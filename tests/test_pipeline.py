"""
Suite de Pruebas Unitarias y de Integración para el Pipeline ETL (Medallion DuckDB + Parquet).
Valida:
- Extract: esquema canónico, validación de Bronze y fallback sintético.
- Transform: limpieza monetaria, features de era y spreads, detección estadística e ingestión DuckDB.
- Load: generación de `silver_cards`, `gold_set_summary`, exportación `.parquet`, esquema sin nulos críticos e idempotencia.
- Analytics: consultas OLAP sobre DuckDB y Parquet.
"""

import os
import uuid
import unittest
import tempfile
import duckdb
import pandas as pd

from src.extract import generate_synthetic_pokemon_data, validate_raw_schema, validate_raw_csvs
from src.transform import clean_currency_series, transform_pokemon_data
from src.load import load_to_duckdb
from src.analytics import get_market_kpis, get_gold_set_summary, get_full_cards_dataset


class TestPokemonETLPipeline(unittest.TestCase):

    def test_01_extract_synthetic_generator(self):
        """Prueba que el generador sintético cree más de 1.500 cartas con el esquema esperado."""
        df = generate_synthetic_pokemon_data(n_cards=1600, seed=123)
        self.assertGreaterEqual(len(df), 1500)
        self.assertTrue(validate_raw_schema(df))
        self.assertGreaterEqual(df["set_id"].nunique(), 15)

        report = validate_raw_csvs("data/raw")
        self.assertIn("csv_count", report)

    def test_02_transform_clean_currency(self):
        """Prueba que la limpieza monetaria maneje formatos sucios, símbolos y espacios."""
        dirty_series = pd.Series(["$12.50", " 1,250.00 ", "$0.99", None, "-5.00", "0.00"])
        cleaned = clean_currency_series(dirty_series)
        self.assertEqual(cleaned.iloc[0], 12.50)
        self.assertEqual(cleaned.iloc[1], 1250.00)
        self.assertEqual(cleaned.iloc[2], 0.99)
        self.assertGreaterEqual(cleaned.min(), 0.01)

    def test_03_transform_statistical_anomalies(self):
        """Prueba que la transformación DuckDB + anomalías genere todas las columnas esperadas."""
        df_raw = generate_synthetic_pokemon_data(n_cards=200, seed=42)
        df_transformed = transform_pokemon_data(df_raw)

        expected_columns = [
            "card_id", "release_date", "snapshot_date", "card_age_years", "era_category",
            "price_spread", "price_spread_pct", "price_tier",
            "is_outlier_iqr", "modified_z_score", "is_outlier_zscore", "is_market_anomaly"
        ]
        for col in expected_columns:
            self.assertIn(col, df_transformed.columns)

        self.assertTrue(pd.api.types.is_bool_dtype(df_transformed["is_market_anomaly"]))
        self.assertTrue(pd.api.types.is_float_dtype(df_transformed["modified_z_score"]))
        self.assertGreater(df_transformed["market_price"].min(), 0.0)

    def test_04_load_duckdb_medallion_and_parquet(self):
        """
        Valida que `silver_cards`, `gold_set_summary` y `cards_clean.parquet`
        se generen con el esquema esperado, sin nulos críticos y de forma idempotente.
        """
        uid = uuid.uuid4().hex
        tmp_db_path = os.path.join(tempfile.gettempdir(), f"test_duckdb_{uid}.duckdb")
        tmp_parquet_path = os.path.join(tempfile.gettempdir(), f"test_silver_{uid}.parquet")

        try:
            df_raw = generate_synthetic_pokemon_data(n_cards=150, seed=99)
            df_clean = transform_pokemon_data(df_raw)

            # Primera carga
            summary_1 = load_to_duckdb(df_clean, db_path=tmp_db_path, parquet_path=tmp_parquet_path)
            self.assertEqual(summary_1["silver_cards"], len(df_clean))
            self.assertGreater(summary_1["gold_set_summary"], 0)
            self.assertTrue(os.path.exists(tmp_parquet_path))
            self.assertGreater(os.path.getsize(tmp_parquet_path), 0)

            # Segunda carga con los mismos datos (idempotencia)
            summary_2 = load_to_duckdb(df_clean, db_path=tmp_db_path, parquet_path=tmp_parquet_path)
            self.assertEqual(summary_2["silver_cards"], summary_1["silver_cards"])
            self.assertEqual(summary_2["gold_set_summary"], summary_1["gold_set_summary"])

            # Validar esquemas y nulos críticos en DuckDB
            conn = duckdb.connect(database=tmp_db_path, read_only=True)
            try:
                silver_cols = set(conn.execute("DESCRIBE silver_cards;").df()["column_name"].tolist())
                gold_cols = set(conn.execute("DESCRIBE gold_set_summary;").df()["column_name"].tolist())

                self.assertTrue({"card_id", "market_price", "release_date", "snapshot_date", "is_market_anomaly"}.issubset(silver_cols))
                self.assertTrue({"set_id", "set_name", "total_cards", "avg_price", "median_price", "max_price", "price_stddev"}.issubset(gold_cols))

                null_silver = conn.execute(
                    "SELECT COUNT(*) FROM silver_cards WHERE card_id IS NULL OR market_price IS NULL OR market_price <= 0;"
                ).fetchone()[0]
                self.assertEqual(null_silver, 0)

                null_gold = conn.execute(
                    "SELECT COUNT(*) FROM gold_set_summary WHERE set_id IS NULL OR total_cards <= 0 OR median_price IS NULL;"
                ).fetchone()[0]
                self.assertEqual(null_gold, 0)
            finally:
                conn.close()

            # Validar lectura desde Parquet y consultas analíticas
            df_parquet = get_full_cards_dataset(db_path=tmp_db_path, parquet_path=tmp_parquet_path)
            self.assertEqual(len(df_parquet), len(df_clean))
            self.assertFalse(df_parquet["card_id"].isna().any())
            self.assertFalse(df_parquet["market_price"].isna().any())

            kpis = get_market_kpis(db_path=tmp_db_path)
            self.assertEqual(kpis["total_cards"], len(df_clean))
            self.assertGreater(kpis["median_price"], 0)

            df_gold = get_gold_set_summary(db_path=tmp_db_path)
            self.assertEqual(len(df_gold), summary_1["gold_set_summary"])
            self.assertIn("price_stddev", df_gold.columns)

        finally:
            for path in (tmp_db_path, tmp_parquet_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
