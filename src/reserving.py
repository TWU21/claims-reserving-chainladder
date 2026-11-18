"""
Métodos de reservas de siniestros sobre triángulos de desarrollo.

- Chain Ladder determinista.
- Modelo de Mack (1993): error cuadrático medio de predicción por año y total.
- Bootstrap ODP (England y Verrall, 2002): distribución completa de la reserva.

Convención: un triángulo es un np.ndarray de tamaño (I, J) con np.nan en las
celdas futuras. Las filas son años de ocurrencia y las columnas, años de desarrollo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- utilidades
def to_cumulative(incremental: np.ndarray) -> np.ndarray:
    """Pasa un triángulo incremental a acumulado (respeta los NaN futuros)."""
    cum = np.nancumsum(incremental, axis=1)
    cum[np.isnan(incremental)] = np.nan
    return cum


def to_incremental(cumulative: np.ndarray) -> np.ndarray:
    """Pasa un triángulo acumulado a incremental."""
    inc = cumulative.copy()
    inc[:, 1:] = cumulative[:, 1:] - cumulative[:, :-1]
    return inc


def latest_diagonal(cumulative: np.ndarray) -> np.ndarray:
    """Último valor observado de cada año de ocurrencia."""
    return np.array([row[~np.isnan(row)][-1] for row in cumulative])


# --------------------------------------------------------------------------- Chain Ladder
def development_factors(cumulative: np.ndarray) -> np.ndarray:
    """Factores de desarrollo ponderados por volumen: f_j = sum C_{i,j+1} / sum C_{i,j}."""
    n_dev = cumulative.shape[1]
    f = np.empty(n_dev - 1)
    for j in range(n_dev - 1):
        mask = ~np.isnan(cumulative[:, j + 1])
        f[j] = cumulative[mask, j + 1].sum() / cumulative[mask, j].sum()
    return f


def project(cumulative: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Completa el triángulo aplicando los factores a las celdas futuras."""
    full = cumulative.copy()
    for j in range(1, full.shape[1]):
        missing = np.isnan(full[:, j])
        full[missing, j] = full[missing, j - 1] * factors[j - 1]
    return full


@dataclass
class ChainLadderResult:
    factors: np.ndarray
    full_triangle: np.ndarray
    latest: np.ndarray
    ultimate: np.ndarray
    reserve: np.ndarray

    def summary(self, origin_labels=None) -> pd.DataFrame:
        idx = origin_labels if origin_labels is not None else range(1, len(self.reserve) + 1)
        df = pd.DataFrame(
            {"Pagado": self.latest, "Último": self.ultimate, "Reserva (IBNR)": self.reserve},
            index=pd.Index(idx, name="Año de ocurrencia"),
        )
        df.loc["Total"] = df.sum()
        return df


def chain_ladder(cumulative: np.ndarray) -> ChainLadderResult:
    f = development_factors(cumulative)
    full = project(cumulative, f)
    latest = latest_diagonal(cumulative)
    ultimate = full[:, -1]
    return ChainLadderResult(f, full, latest, ultimate, ultimate - latest)


# --------------------------------------------------------------------------- Mack
@dataclass
class MackResult:
    chain_ladder: ChainLadderResult
    sigma2: np.ndarray
    se_by_origin: np.ndarray
    se_total: float

    def summary(self, origin_labels=None) -> pd.DataFrame:
        df = self.chain_ladder.summary(origin_labels)
        se = np.append(self.se_by_origin, self.se_total)
        df["Error estándar (Mack)"] = se
        df["CV"] = df["Error estándar (Mack)"] / df["Reserva (IBNR)"].replace(0, np.nan)
        return df


def mack(cumulative: np.ndarray) -> MackResult:
    """Modelo de Mack (1993), independiente de la distribución.

    Devuelve el error estándar de predicción (raíz del MSEP) de la reserva de cada
    año de ocurrencia y del total.
    """
    cl = chain_ladder(cumulative)
    f, full = cl.factors, cl.full_triangle
    n_orig, n_dev = cumulative.shape

    # Varianzas sigma_j^2
    sigma2 = np.empty(n_dev - 1)
    for j in range(n_dev - 1):
        mask = ~np.isnan(cumulative[:, j + 1])
        c_j = cumulative[mask, j]
        ratios = cumulative[mask, j + 1] / c_j
        n_obs = mask.sum()
        if n_obs > 1:
            sigma2[j] = np.sum(c_j * (ratios - f[j]) ** 2) / (n_obs - 1)
        else:  # último factor: extrapolación de Mack
            sigma2[j] = min(sigma2[j - 1] ** 2 / sigma2[j - 2], sigma2[j - 2], sigma2[j - 1])

    # Suma de C_{.,j} sobre los años con el siguiente desarrollo observado
    col_sums = np.array(
        [cumulative[~np.isnan(cumulative[:, j + 1]), j].sum() for j in range(n_dev - 1)]
    )

    # Primer desarrollo no observado de cada año de ocurrencia
    first_missing = np.array([np.sum(~np.isnan(row)) for row in cumulative])

    mse = np.zeros(n_orig)
    for i in range(n_orig):
        k_range = range(first_missing[i] - 1, n_dev - 1)
        mse[i] = full[i, -1] ** 2 * sum(
            sigma2[k] / f[k] ** 2 * (1 / full[i, k] + 1 / col_sums[k]) for k in k_range
        )

    total_mse = mse.sum()
    for i in range(n_orig):
        later_ultimates = full[i + 1 :, -1].sum()
        k_range = range(first_missing[i] - 1, n_dev - 1)
        total_mse += full[i, -1] * later_ultimates * sum(
            2 * sigma2[k] / f[k] ** 2 / col_sums[k] for k in k_range
        )

    return MackResult(cl, sigma2, np.sqrt(mse), float(np.sqrt(total_mse)))


# --------------------------------------------------------------------------- Bootstrap ODP
@dataclass
class BootstrapResult:
    reserves_by_origin: np.ndarray  # (n_sims, n_orig)
    total: np.ndarray  # (n_sims,)
    phi: float

    def percentiles(self, qs=(0.5, 0.75, 0.95, 0.995)) -> pd.Series:
        return pd.Series(np.quantile(self.total, qs), index=[f"P{q * 100:g}" for q in qs])


def _fitted_incremental(cumulative: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Incrementales ajustados por Chain Ladder hacia atrás desde la última diagonal."""
    n_orig, n_dev = cumulative.shape
    fitted_cum = np.full_like(cumulative, np.nan)
    latest = latest_diagonal(cumulative)
    for i in range(n_orig):
        last = np.sum(~np.isnan(cumulative[i])) - 1
        fitted_cum[i, last] = latest[i]
        for j in range(last - 1, -1, -1):
            fitted_cum[i, j] = fitted_cum[i, j + 1] / factors[j]
    return to_incremental(fitted_cum)


def bootstrap_odp(
    cumulative: np.ndarray, n_sims: int = 10_000, seed: int = 2026
) -> BootstrapResult:
    """Bootstrap del modelo de Poisson sobredisperso (England y Verrall, 2002).

    1. Ajusta Chain Ladder y calcula los residuos de Pearson ajustados.
    2. Remuestrea residuos para crear triángulos ficticios y reestima los factores
       (incertidumbre de estimación).
    3. Simula los pagos futuros con una Gamma de media m y varianza phi*m
       (incertidumbre de proceso).
    """
    rng = np.random.default_rng(seed)
    n_orig, n_dev = cumulative.shape
    incremental = to_incremental(cumulative)
    f = development_factors(cumulative)
    m = _fitted_incremental(cumulative, f)

    observed = ~np.isnan(incremental)
    resid = (incremental[observed] - m[observed]) / np.sqrt(m[observed])
    n_obs = observed.sum()
    n_par = n_orig + n_dev - 1
    phi = float(np.sum(resid**2) / (n_obs - n_par))
    adj_resid = resid * np.sqrt(n_obs / (n_obs - n_par))
    # Las esquinas tienen residuo 0 por construcción: se excluyen del remuestreo
    pool = adj_resid[np.abs(adj_resid) > 1e-12]

    future = np.isnan(cumulative)
    reserves = np.empty((n_sims, n_orig))
    for s in range(n_sims):
        pseudo_inc = np.full_like(incremental, np.nan)
        pseudo_inc[observed] = m[observed] + rng.choice(pool, n_obs) * np.sqrt(m[observed])
        pseudo_cum = to_cumulative(pseudo_inc)
        f_star = development_factors(pseudo_cum)
        full_star = project(pseudo_cum, f_star)  # proyección desde el triángulo ficticio
        inc_star = to_incremental(full_star)
        mean_future = np.where(future, np.clip(inc_star, 1e-9, None), 0.0)
        # Gamma con media mu y varianza phi*mu: shape = mu/phi, scale = phi
        sims = np.where(future, rng.gamma(mean_future / phi, phi), 0.0)
        reserves[s] = sims.sum(axis=1)

    return BootstrapResult(reserves, reserves.sum(axis=1), phi)
