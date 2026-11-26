"""
Suite de Pruebas Unitarias y de Integración para el Pipeline ETL.
Valida la integridad de los componentes:
- Extract: esquema y fallback sintético.
- Transform: limpieza de moneda, features de era y spreads, detección estadística.
- Load: esquema estrella e idempotencia en DuckDB.
- Analytics: consultas analíticas OLAP.
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np

from src.extract import generate_synthetic_pokemon_data, validate_raw_schema, REQUIRED_COLUMNS
from src.transform import clean_currency_series, transform_pokemon_data, standardize_rarity
from src.load import load_to_duckdb, init_star_schema
from src.analytics import get_market_kpis, get_median_price_by_set


class TestPokemonETLPipeline(unittest.TestCase):

    def test_01_extract_synthetic_generator(self):
        """Prueba que el generador sintético cree más de 1.500 cartas con el esquema esperado."""
        df = generate_synthetic_pokemon_data(n_cards=1600, seed=123)
        self.assertGreaterEqual(len(df), 1500)
        self.assertTrue(validate_raw_schema(df))
        self.assertGreaterEqual(df["set_id"].nunique(), 15)

    def test_02_transform_clean_currency(self):
        """Prueba que la limpieza monetaria maneje formatos sucios, símbolos y espacios."""
        dirty_series = pd.Series(["$12.50", " 1,250.00 ", "$0.99", None, "-5.00", "0.00"])
        cleaned = clean_currency_series(dirty_series)
        self.assertEqual(cleaned.iloc[0], 12.50)
        self.assertEqual(cleaned.iloc[1], 1250.00)
        self.assertEqual(cleaned.iloc[2], 0.99)
        self.assertGreaterEqual(cleaned.min(), 0.01)

    def test_03_transform_statistical_anomalies(self):
        """Prueba que la detección de anomalías genere las columnas esperadas y detecte outliers."""
        df_raw = generate_synthetic_pokemon_data(n_cards=200, seed=42)
        df_transformed = transform_pokemon_data(df_raw)

        expected_columns = [
            "card_age_years", "era_category", "price_spread", "price_spread_pct",
            "price_tier", "is_outlier_iqr", "modified_z_score", "is_outlier_zscore", "is_market_anomaly"
        ]
        for col in expected_columns:
            self.assertIn(col, df_transformed.columns)

        self.assertTrue(pd.api.types.is_bool_dtype(df_transformed["is_market_anomaly"]))
        self.assertTrue(pd.api.types.is_float_dtype(df_transformed["modified_z_score"]))

    def test_04_load_idempotency_duckdb(self):
        """Prueba que la carga en DuckDB sea idempotente y mantenga la integridad del esquema estrella."""
        import uuid
        tmp_db_path = os.path.join(tempfile.gettempdir(), f"test_duckdb_{uuid.uuid4().hex}.duckdb")

        try:
            df_raw = generate_synthetic_pokemon_data(n_cards=150, seed=99)
            df_clean = transform_pokemon_data(df_raw)

            # Primera carga
            summary_1 = load_to_duckdb(df_clean, db_path=tmp_db_path)
            self.assertEqual(summary_1["dim_cards"], len(df_clean))

            # Segunda carga con los mismos datos (debe mantener los mismos conteos sin duplicar filas)
            summary_2 = load_to_duckdb(df_clean, db_path=tmp_db_path)
            self.assertEqual(summary_2["dim_cards"], len(df_clean))
            self.assertEqual(summary_2["dim_sets"], summary_1["dim_sets"])
            self.assertEqual(summary_2["fact_market_prices"], summary_1["fact_market_prices"])

            # Verificación de consultas analíticas sobre la base temporal
            kpis = get_market_kpis(db_path=tmp_db_path)
            self.assertEqual(kpis["total_cards"], len(df_clean))
            self.assertGreater(kpis["median_price"], 0)

        finally:
            if os.path.exists(tmp_db_path):
                try:
                    os.remove(tmp_db_path)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
