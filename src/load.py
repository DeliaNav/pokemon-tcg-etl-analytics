"""
Módulo de Carga (Capas Silver & Gold) para el pipeline ETL de Pokémon TCG.

Persiste el Data Warehouse en DuckDB (`data/warehouse/pokemon_tcg.duckdb`) y exporta
la capa Silver en formato columnar Parquet (`data/processed/cards_clean.parquet`):
- Capa Silver (`silver_cards`): Datos limpios, fechas normalizadas, precios DOUBLE,
  features analíticas y banderas de anomalías estadísticas (deduplicados por carta/snapshot).
- Capa Gold (`gold_set_summary`): Agregación por expansión/set con total de cartas,
  precio medio, mediana, mínimo, máximo, volatilidad (desviación estándar) y anomalías.
"""

import os
import logging
from typing import Dict
import duckdb
import pandas as pd

# Configuración del registrador de logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [LOAD]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/warehouse/pokemon_tcg.duckdb"
DEFAULT_PARQUET_PATH = "data/processed/cards_clean.parquet"


def init_medallion_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Inicializa las tablas de la arquitectura Medallion (Silver y Gold) en DuckDB.
    """
    logger.info("Inicializando DDL de Capas Silver y Gold en DuckDB...")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS silver_cards (
        card_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        supertype VARCHAR NOT NULL,
        subtype VARCHAR,
        rarity VARCHAR NOT NULL,
        rarity_group VARCHAR NOT NULL,
        artist VARCHAR,
        set_id VARCHAR NOT NULL,
        set_name VARCHAR NOT NULL,
        series VARCHAR NOT NULL,
        release_date DATE NOT NULL,
        era_category VARCHAR NOT NULL,
        market_price DOUBLE NOT NULL,
        low_price DOUBLE NOT NULL,
        high_price DOUBLE NOT NULL,
        price_spread DOUBLE NOT NULL,
        price_spread_pct DOUBLE NOT NULL,
        price_tier VARCHAR NOT NULL,
        card_age_years DOUBLE NOT NULL,
        modified_z_score DOUBLE NOT NULL,
        is_outlier_iqr BOOLEAN NOT NULL,
        is_outlier_zscore BOOLEAN NOT NULL,
        is_market_anomaly BOOLEAN NOT NULL,
        snapshot_date DATE NOT NULL,
        load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS gold_set_summary (
        set_id VARCHAR PRIMARY KEY,
        set_name VARCHAR NOT NULL,
        series VARCHAR NOT NULL,
        release_date DATE NOT NULL,
        era_category VARCHAR NOT NULL,
        total_cards INTEGER NOT NULL,
        avg_price DOUBLE NOT NULL,
        median_price DOUBLE NOT NULL,
        min_price DOUBLE NOT NULL,
        max_price DOUBLE NOT NULL,
        price_stddev DOUBLE NOT NULL,
        avg_spread_pct DOUBLE NOT NULL,
        anomaly_count INTEGER NOT NULL,
        load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    logger.info("Esquema Medallion inicializado correctamente (silver_cards, gold_set_summary).")


def load_to_duckdb(
    df: pd.DataFrame,
    db_path: str = DEFAULT_DB_PATH,
    parquet_path: str = DEFAULT_PARQUET_PATH,
) -> Dict[str, int]:
    """
    Carga el DataFrame transformado en `silver_cards` (DuckDB), lo exporta a Parquet
    (`data/processed/cards_clean.parquet`) y construye la tabla agregada `gold_set_summary`.
    Garantiza idempotencia mediante transacciones atómicas y upserts / reemplazo controlado.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)

    logger.info(f"Conectando a DuckDB Warehouse: '{db_path}'...")
    conn = duckdb.connect(database=db_path, read_only=False)

    try:
        conn.begin()
        init_medallion_schema(conn)

        # Preparar columnas de staging para silver_cards
        silver_cols = [
            "card_id", "name", "supertype", "subtype", "rarity", "rarity_group", "artist",
            "set_id", "set_name", "series", "release_date", "era_category",
            "market_price", "low_price", "high_price", "price_spread", "price_spread_pct",
            "price_tier", "card_age_years", "modified_z_score",
            "is_outlier_iqr", "is_outlier_zscore", "is_market_anomaly", "snapshot_date"
        ]
        df_silver = df[silver_cols].drop_duplicates(subset=["card_id"], keep="last").copy()

        conn.register("staging_silver", df_silver)
        conn.execute("""
        INSERT INTO silver_cards (
            card_id, name, supertype, subtype, rarity, rarity_group, artist,
            set_id, set_name, series, release_date, era_category,
            market_price, low_price, high_price, price_spread, price_spread_pct,
            price_tier, card_age_years, modified_z_score,
            is_outlier_iqr, is_outlier_zscore, is_market_anomaly, snapshot_date, load_timestamp
        )
        SELECT
            card_id,
            name,
            supertype,
            subtype,
            rarity,
            rarity_group,
            artist,
            set_id,
            set_name,
            series,
            CAST(release_date AS DATE),
            era_category,
            CAST(market_price AS DOUBLE),
            CAST(low_price AS DOUBLE),
            CAST(high_price AS DOUBLE),
            CAST(price_spread AS DOUBLE),
            CAST(price_spread_pct AS DOUBLE),
            price_tier,
            CAST(card_age_years AS DOUBLE),
            CAST(modified_z_score AS DOUBLE),
            CAST(is_outlier_iqr AS BOOLEAN),
            CAST(is_outlier_zscore AS BOOLEAN),
            CAST(is_market_anomaly AS BOOLEAN),
            CAST(snapshot_date AS DATE),
            now()
        FROM staging_silver
        ON CONFLICT (card_id) DO UPDATE SET
            name = EXCLUDED.name,
            supertype = EXCLUDED.supertype,
            subtype = EXCLUDED.subtype,
            rarity = EXCLUDED.rarity,
            rarity_group = EXCLUDED.rarity_group,
            artist = EXCLUDED.artist,
            set_id = EXCLUDED.set_id,
            set_name = EXCLUDED.set_name,
            series = EXCLUDED.series,
            release_date = EXCLUDED.release_date,
            era_category = EXCLUDED.era_category,
            market_price = EXCLUDED.market_price,
            low_price = EXCLUDED.low_price,
            high_price = EXCLUDED.high_price,
            price_spread = EXCLUDED.price_spread,
            price_spread_pct = EXCLUDED.price_spread_pct,
            price_tier = EXCLUDED.price_tier,
            card_age_years = EXCLUDED.card_age_years,
            modified_z_score = EXCLUDED.modified_z_score,
            is_outlier_iqr = EXCLUDED.is_outlier_iqr,
            is_outlier_zscore = EXCLUDED.is_outlier_zscore,
            is_market_anomaly = EXCLUDED.is_market_anomaly,
            snapshot_date = EXCLUDED.snapshot_date,
            load_timestamp = now();
        """)
        conn.unregister("staging_silver")

        # Construcción de Capa Gold: Agregación por expansión/set
        conn.execute("DELETE FROM gold_set_summary;")
        conn.execute("""
        INSERT INTO gold_set_summary (
            set_id, set_name, series, release_date, era_category,
            total_cards, avg_price, median_price, min_price, max_price,
            price_stddev, avg_spread_pct, anomaly_count, load_timestamp
        )
        SELECT
            set_id,
            ANY_VALUE(set_name) AS set_name,
            ANY_VALUE(series) AS series,
            MIN(release_date) AS release_date,
            ANY_VALUE(era_category) AS era_category,
            COUNT(*)::INTEGER AS total_cards,
            ROUND(AVG(market_price), 2) AS avg_price,
            ROUND(MEDIAN(market_price), 2) AS median_price,
            ROUND(MIN(market_price), 2) AS min_price,
            ROUND(MAX(market_price), 2) AS max_price,
            ROUND(COALESCE(STDDEV_SAMP(market_price), 0.0), 2) AS price_stddev,
            ROUND(AVG(price_spread_pct), 2) AS avg_spread_pct,
            SUM(CASE WHEN is_market_anomaly THEN 1 ELSE 0 END)::INTEGER AS anomaly_count,
            now()
        FROM silver_cards
        GROUP BY set_id;
        """)

        # Exportación columnar Parquet de silver_cards
        parquet_sql_path = parquet_path.replace("\\", "/")
        conn.execute(f"COPY silver_cards TO '{parquet_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")

        conn.commit()

        count_silver = conn.execute("SELECT COUNT(*) FROM silver_cards;").fetchone()[0]
        count_gold = conn.execute("SELECT COUNT(*) FROM gold_set_summary;").fetchone()[0]

        summary = {
            "silver_cards": int(count_silver),
            "gold_set_summary": int(count_gold),
        }
        logger.info(f"Carga Medallion completada. DuckDB: {summary} | Parquet exportado en '{parquet_path}'.")
        return summary

    except Exception as ex:
        conn.rollback()
        logger.error(f"Error crítico durante la carga en DuckDB/Parquet: {ex}")
        raise ex
    finally:
        conn.close()
        logger.info("Conexión con DuckDB cerrada correctamente.")


if __name__ == "__main__":
    from extract import extract_raw_data
    from transform import transform_pokemon_data

    raw = extract_raw_data()
    clean = transform_pokemon_data(raw)
    load_to_duckdb(clean)
