"""
Módulo Analítico para consultas OLAP sobre DuckDB y Parquet (Capas Silver y Gold).

Provee consultas analíticas de alto rendimiento sobre:
- `silver_cards` (y `data/processed/cards_clean.parquet`)
- `gold_set_summary`
"""

import os
import logging
from typing import Dict, Any
import duckdb
import pandas as pd

# Configuración del registrador de logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [ANALYTICS]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/warehouse/pokemon_tcg.duckdb"
DEFAULT_PARQUET_PATH = "data/processed/cards_clean.parquet"


def get_connection(db_path: str = DEFAULT_DB_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Establece una conexión optimizada a DuckDB en modo solo lectura."""
    return duckdb.connect(database=db_path, read_only=read_only)


def get_market_kpis(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Obtiene los principales Indicadores Clave de Desempeño (KPIs) desde `silver_cards`."""
    conn = get_connection(db_path)
    try:
        query = """
        SELECT
            COUNT(card_id) AS total_cards,
            COUNT(DISTINCT set_id) AS total_sets,
            COUNT(DISTINCT series) AS total_series,
            ROUND(AVG(market_price), 2) AS avg_price,
            ROUND(MEDIAN(market_price), 2) AS median_price,
            ROUND(MAX(market_price), 2) AS max_price,
            ROUND(SUM(market_price), 2) AS estimated_market_cap,
            SUM(CASE WHEN is_market_anomaly THEN 1 ELSE 0 END) AS total_anomalies,
            ROUND(100.0 * SUM(CASE WHEN is_market_anomaly THEN 1 ELSE 0 END) / COUNT(card_id), 2) AS anomaly_rate_pct,
            ROUND(AVG(price_spread_pct), 2) AS avg_spread_pct
        FROM silver_cards;
        """
        row = conn.execute(query).df().iloc[0].to_dict()

        top_card_query = """
        SELECT name, set_name, market_price
        FROM silver_cards
        ORDER BY market_price DESC
        LIMIT 1;
        """
        top_card = conn.execute(top_card_query).df()
        if not top_card.empty:
            row["top_card_name"] = top_card.iloc[0]["name"]
            row["top_card_set"] = top_card.iloc[0]["set_name"]
            row["top_card_price"] = top_card.iloc[0]["market_price"]
        else:
            row["top_card_name"] = "N/A"
            row["top_card_set"] = "N/A"
            row["top_card_price"] = 0.0

        return row
    finally:
        conn.close()


def get_gold_set_summary(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Consulta directamente la tabla agregada `gold_set_summary` ordenada por mediana de precio.
    Incluye métricas de volatilidad (`price_stddev`), media, mediana, mínimo y máximo.
    """
    conn = get_connection(db_path)
    try:
        query = """
        SELECT
            set_id,
            set_name,
            series,
            era_category,
            release_date,
            total_cards,
            median_price,
            avg_price,
            min_price,
            max_price,
            price_stddev,
            avg_spread_pct,
            anomaly_count
        FROM gold_set_summary
        ORDER BY median_price DESC;
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_median_price_by_set(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retorna métricas por set desde `gold_set_summary` manteniendo compatibilidad de columnas.
    """
    df_gold = get_gold_set_summary(db_path=db_path)
    return df_gold.rename(
        columns={
            "total_cards": "cards_tracked",
            "median_price": "median_market_price",
            "avg_price": "avg_market_price",
            "min_price": "min_market_price",
            "max_price": "max_market_price",
        }
    )


def get_top_anomalies(db_path: str = DEFAULT_DB_PATH, limit: int = 30) -> pd.DataFrame:
    """Extrae las cartas identificadas como anomalías estadísticas desde `silver_cards`."""
    conn = get_connection(db_path)
    try:
        query = f"""
        SELECT
            card_id,
            name,
            supertype,
            subtype,
            rarity,
            rarity_group,
            set_name,
            series,
            era_category,
            artist,
            market_price,
            low_price,
            high_price,
            price_spread,
            price_spread_pct,
            modified_z_score,
            price_tier
        FROM silver_cards
        WHERE is_market_anomaly = TRUE
        ORDER BY modified_z_score DESC, market_price DESC
        LIMIT {int(limit)};
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_vintage_vs_modern_evolution(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Compara métricas macroeconómicas entre eras históricas desde `silver_cards`."""
    conn = get_connection(db_path)
    try:
        query = """
        SELECT
            era_category,
            COUNT(card_id) AS total_cards,
            COUNT(DISTINCT set_id) AS total_sets,
            ROUND(MEDIAN(market_price), 2) AS median_price,
            ROUND(AVG(market_price), 2) AS avg_price,
            ROUND(MAX(market_price), 2) AS max_price,
            ROUND(AVG(price_spread_pct), 2) AS avg_spread_pct,
            SUM(CASE WHEN price_tier = 'Grail (>$100)' THEN 1 ELSE 0 END) AS grail_cards_count,
            SUM(CASE WHEN is_market_anomaly THEN 1 ELSE 0 END) AS anomaly_cards_count,
            ROUND(100.0 * SUM(CASE WHEN is_market_anomaly THEN 1 ELSE 0 END) / COUNT(card_id), 2) AS anomaly_percentage
        FROM silver_cards
        GROUP BY era_category
        ORDER BY median_price DESC;
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_rarity_price_distribution(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Obtiene la distribución cuantitativa de precios por grupo de rareza."""
    conn = get_connection(db_path)
    try:
        query = """
        SELECT
            rarity_group,
            COUNT(card_id) AS card_count,
            ROUND(MIN(market_price), 2) AS min_price,
            ROUND(QUANTILE_CONT(market_price, 0.25), 2) AS p25_price,
            ROUND(MEDIAN(market_price), 2) AS median_price,
            ROUND(QUANTILE_CONT(market_price, 0.75), 2) AS p75_price,
            ROUND(MAX(market_price), 2) AS max_price,
            ROUND(AVG(market_price), 2) AS avg_price,
            SUM(CASE WHEN is_market_anomaly THEN 1 ELSE 0 END) AS anomalies_count
        FROM silver_cards
        GROUP BY rarity_group
        ORDER BY median_price DESC;
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_top_artists_by_market_value(
    db_path: str = DEFAULT_DB_PATH, min_cards: int = 5, limit: int = 15
) -> pd.DataFrame:
    """Identifica los ilustradores cuyas cartas alcanzan mayor valoración mediana."""
    conn = get_connection(db_path)
    try:
        query = f"""
        SELECT
            artist,
            COUNT(card_id) AS total_illustrations,
            ROUND(MEDIAN(market_price), 2) AS median_card_price,
            ROUND(AVG(market_price), 2) AS avg_card_price,
            ROUND(MAX(market_price), 2) AS most_expensive_card_price,
            SUM(CASE WHEN price_tier = 'Grail (>$100)' THEN 1 ELSE 0 END) AS grails_count
        FROM silver_cards
        WHERE artist IS NOT NULL AND artist != 'Desconocido'
        GROUP BY artist
        HAVING COUNT(card_id) >= {int(min_cards)}
        ORDER BY median_card_price DESC, avg_card_price DESC
        LIMIT {int(limit)};
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_full_cards_dataset(
    db_path: str = DEFAULT_DB_PATH,
    parquet_path: str = DEFAULT_PARQUET_PATH,
) -> pd.DataFrame:
    """
    Retorna la vista completa Silver leyendo directamente desde Parquet columnar
    cuando está disponible (o desde DuckDB `silver_cards` como respaldo).
    """
    if os.path.exists(parquet_path):
        parquet_sql_path = parquet_path.replace("\\", "/")
        return duckdb.sql(f"SELECT * FROM read_parquet('{parquet_sql_path}');").df()

    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT * FROM silver_cards;").df()
    finally:
        conn.close()


def execute_custom_query(query: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Permite ejecutar consultas SQL personalizadas de solo lectura sobre DuckDB."""
    conn = get_connection(db_path)
    try:
        return conn.execute(query).df()
    finally:
        conn.close()


if __name__ == "__main__":
    kpis = get_market_kpis()
    print("\n--- KPIs de Mercado (Gold/Silver) ---")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
