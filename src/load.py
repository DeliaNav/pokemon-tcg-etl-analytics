"""
Módulo de Carga para el pipeline ETL de Pokémon TCG.

Implementa un Esquema Estrella (Star Schema) en DuckDB:
- Tabla Dimensional: dim_sets (Metadatos del set, serie, fecha de lanzamiento, era).
- Tabla Dimensional: dim_cards (Identidad de la carta, supertipo, subtipo, rareza, artista).
- Tabla de Hechos: fact_market_prices (Métricas monetarias, spreads, anomalías estadísticas, timestamp).

Garantiza inserción 100% idempotente mediante transacciones atómicas y
estrategia de upsert (ON CONFLICT DO UPDATE).
"""

import os
import logging
from typing import Optional
import duckdb
import pandas as pd

# Configuración del registrador de logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [LOAD]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/pokemon_market.duckdb"


def init_star_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Crea las tablas dimensionales y la tabla de hechos del esquema estrella en DuckDB.
    En DuckDB (motor OLAP columnar de alto rendimiento), las relaciones dimensionales
    se modelan con PRIMARY KEYs e índices lógicos en las claves foráneas para maximizar
    el rendimiento vectorial en joins y garantizar upserts transaccionales idempotentes.
    """
    logger.info("Inicializando DDL del Esquema Estrella en DuckDB...")

    # 1. Dimensión de Sets
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dim_sets (
        set_id VARCHAR PRIMARY KEY,
        set_name VARCHAR NOT NULL,
        series VARCHAR NOT NULL,
        release_date DATE NOT NULL,
        era_category VARCHAR NOT NULL,
        total_cards_count INTEGER NOT NULL
    );
    """)

    # 2. Dimensión de Cartas
    conn.execute("""
    CREATE TABLE IF NOT EXISTS dim_cards (
        card_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        supertype VARCHAR NOT NULL,
        subtype VARCHAR,
        rarity VARCHAR NOT NULL,
        rarity_group VARCHAR NOT NULL,
        artist VARCHAR,
        set_id VARCHAR NOT NULL
    );
    """)

    # 3. Tabla de Hechos de Precios de Mercado
    conn.execute("""
    CREATE TABLE IF NOT EXISTS fact_market_prices (
        fact_id VARCHAR PRIMARY KEY,
        card_id VARCHAR NOT NULL,
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
        load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    logger.info("Esquema estrella inicializado correctamente (dim_sets, dim_cards, fact_market_prices).")


def load_to_duckdb(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> dict:
    """
    Carga el DataFrame transformado en las tablas del esquema estrella de DuckDB
    de forma transaccional e idempotente.

    Args:
        df: DataFrame procesado por el módulo transform.
        db_path: Ruta al archivo DuckDB.

    Returns:
        Diccionario con conteos de registros cargados en cada tabla.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    logger.info(f"Conectando a base de datos DuckDB: {db_path}...")

    conn = duckdb.connect(database=db_path, read_only=False)

    try:
        # Iniciar transacción
        conn.begin()

        # Asegurar esquema
        init_star_schema(conn)

        # -------------------------------------------------------------
        # 1. Preparar y cargar DIM_SETS
        # -------------------------------------------------------------
        set_counts = df.groupby("set_id").size().to_dict()
        df_sets = (
            df[["set_id", "set_name", "series", "release_date", "era_category"]]
            .drop_duplicates(subset=["set_id"])
            .copy()
        )
        df_sets["total_cards_count"] = df_sets["set_id"].map(set_counts).fillna(0).astype(int)

        # Carga idempotente en dim_sets
        conn.register("staging_sets", df_sets)
        conn.execute("""
        INSERT INTO dim_sets
        SELECT 
            set_id,
            set_name,
            series,
            CAST(release_date AS DATE),
            era_category,
            total_cards_count
        FROM staging_sets
        ON CONFLICT (set_id) DO UPDATE SET
            set_name = EXCLUDED.set_name,
            series = EXCLUDED.series,
            release_date = EXCLUDED.release_date,
            era_category = EXCLUDED.era_category,
            total_cards_count = EXCLUDED.total_cards_count;
        """)
        conn.unregister("staging_sets")

        # -------------------------------------------------------------
        # 2. Preparar y cargar DIM_CARDS
        # -------------------------------------------------------------
        df_cards = (
            df[["card_id", "name", "supertype", "subtype", "rarity", "rarity_group", "artist", "set_id"]]
            .drop_duplicates(subset=["card_id"])
            .copy()
        )

        conn.register("staging_cards", df_cards)
        conn.execute("""
        INSERT INTO dim_cards
        SELECT 
            card_id,
            name,
            supertype,
            subtype,
            rarity,
            rarity_group,
            artist,
            set_id
        FROM staging_cards
        ON CONFLICT (card_id) DO UPDATE SET
            name = EXCLUDED.name,
            supertype = EXCLUDED.supertype,
            subtype = EXCLUDED.subtype,
            rarity = EXCLUDED.rarity,
            rarity_group = EXCLUDED.rarity_group,
            artist = EXCLUDED.artist,
            set_id = EXCLUDED.set_id;
        """)
        conn.unregister("staging_cards")

        # -------------------------------------------------------------
        # 3. Preparar y cargar FACT_MARKET_PRICES
        # -------------------------------------------------------------
        df_facts = df[[
            "card_id", "market_price", "low_price", "high_price",
            "price_spread", "price_spread_pct", "price_tier", "card_age_years",
            "modified_z_score", "is_outlier_iqr", "is_outlier_zscore", "is_market_anomaly"
        ]].copy()
        
        # Generar clave primaria unívoca fact_id (idempotente vinculada a card_id)
        df_facts["fact_id"] = df_facts["card_id"]

        conn.register("staging_facts", df_facts)
        conn.execute("""
        INSERT INTO fact_market_prices (
            fact_id,
            card_id,
            market_price,
            low_price,
            high_price,
            price_spread,
            price_spread_pct,
            price_tier,
            card_age_years,
            modified_z_score,
            is_outlier_iqr,
            is_outlier_zscore,
            is_market_anomaly,
            load_timestamp
        )
        SELECT 
            fact_id,
            card_id,
            market_price,
            low_price,
            high_price,
            price_spread,
            price_spread_pct,
            price_tier,
            card_age_years,
            modified_z_score,
            is_outlier_iqr,
            is_outlier_zscore,
            is_market_anomaly,
            now()
        FROM staging_facts
        ON CONFLICT (fact_id) DO UPDATE SET
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
            load_timestamp = now();
        """)
        conn.unregister("staging_facts")

        # Confirmar transacción
        conn.commit()

        # Verificación y conteos
        count_sets = conn.execute("SELECT COUNT(*) FROM dim_sets;").fetchone()[0]
        count_cards = conn.execute("SELECT COUNT(*) FROM dim_cards;").fetchone()[0]
        count_facts = conn.execute("SELECT COUNT(*) FROM fact_market_prices;").fetchone()[0]

        summary = {
            "dim_sets": count_sets,
            "dim_cards": count_cards,
            "fact_market_prices": count_facts
        }
        logger.info(f"Carga en DuckDB completada exitosamente. Registros: {summary}")
        return summary

    except Exception as ex:
        conn.rollback()
        logger.error(f"Error crítico durante la carga en DuckDB: {ex}")
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
