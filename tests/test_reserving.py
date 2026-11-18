"""Comprueba la implementación con los resultados publicados por Mack (1993)
para el triángulo de Taylor y Ashe (1983).

Ejecutar desde la raíz del repositorio:  python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reserving import bootstrap_odp, chain_ladder, mack, to_cumulative, to_incremental  # noqa: E402


def load_triangle() -> np.ndarray:
    return pd.read_csv(ROOT / "data" / "taylor_ashe_cumulative.csv", index_col=0).to_numpy(float)


class TestChainLadder(unittest.TestCase):
    def setUp(self):
        self.tri = load_triangle()

    def test_development_factors(self):
        published = [3.4906, 1.7473, 1.4574, 1.1739, 1.1038, 1.0863, 1.0539, 1.0766, 1.0177]
        np.testing.assert_allclose(chain_ladder(self.tri).factors, published, atol=1e-4)

    def test_total_reserve(self):
        # Mack (1993): reserva total de 18.680.856
        self.assertAlmostEqual(chain_ladder(self.tri).reserve.sum(), 18_680_856, delta=5)

    def test_incremental_roundtrip(self):
        np.testing.assert_allclose(to_cumulative(to_incremental(self.tri)), self.tri)


class TestMack(unittest.TestCase):
    def test_total_standard_error(self):
        # Mack (1993): error estándar total de 2.447.095 (CV = 13 %)
        self.assertAlmostEqual(mack(load_triangle()).se_total, 2_447_095, delta=1_000)


class TestBootstrap(unittest.TestCase):
    def test_bootstrap_mean_close_to_chain_ladder(self):
        res = bootstrap_odp(load_triangle(), n_sims=2_000, seed=1)
        self.assertAlmostEqual(res.total.mean() / 18_680_856, 1.0, delta=0.03)


if __name__ == "__main__":
    unittest.main()
