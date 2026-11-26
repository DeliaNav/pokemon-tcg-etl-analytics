"""
Módulo Analítico para consultas OLAP sobre DuckDB.

Contiene funciones especializadas para la explotación de datos analíticos:
- KPIs globales de mercado.
- Medianas y dispersión de precios por Set y Serie.
- Ranking de anomalías y cartas de alta valorización (Grails / Spikes).
- Evolución comparativa Vintage vs Moderno.
- Distribución de precios por Rareza y Supertipo.
- Métricas de artistas más cotizados.
- Ejecutor de consultas SQL arbitrarias para exploración libre.
"""

import logging
from typing import Dict, Any, Optional
import duckdb
import pandas as pd

# Configuración del registrador de logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [ANALYTICS]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/pokemon_market.duckdb"


def get_connection(db_path: str = DEFAULT_DB_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Establece una conexión optimizada a DuckDB en modo solo lectura para análisis concurrentes.
    """
    return duckdb.connect(database=db_path, read_only=read_only)


def get_market_kpis(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Obtiene los principales Indicadores Clave de Desempeño (KPIs) del mercado.
    """
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            COUNT(c.card_id) AS total_cards,
            COUNT(DISTINCT s.set_id) AS total_sets,
            COUNT(DISTINCT s.series) AS total_series,
            ROUND(AVG(f.market_price), 2) AS avg_price,
            ROUND(MEDIAN(f.market_price), 2) AS median_price,
            ROUND(MAX(f.market_price), 2) AS max_price,
            ROUND(SUM(f.market_price), 2) AS estimated_market_cap,
            SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) AS total_anomalies,
            ROUND(100.0 * SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) / COUNT(c.card_id), 2) AS anomaly_rate_pct,
            ROUND(AVG(f.price_spread_pct), 2) AS avg_spread_pct
        FROM fact_market_prices f
        JOIN dim_cards c ON f.card_id = c.card_id
        JOIN dim_sets s ON c.set_id = s.set_id;
        """
        row = conn.execute(query).df().iloc[0].to_dict()

        # Identificar la carta más valiosa
        top_card_query = """
        SELECT c.name, s.set_name, f.market_price
        FROM fact_market_prices f
        JOIN dim_cards c ON f.card_id = c.card_id
        JOIN dim_sets s ON c.set_id = s.set_id
        ORDER BY f.market_price DESC
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


def get_median_price_by_set(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Calcula estadísticas agregadas de precios (mediana, promedio, mín, máx, spreads y anomalías)
    agrupadas por Set, ordenadas descendentemente por precio mediano.
    """
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            s.set_id,
            s.set_name,
            s.series,
            s.era_category,
            s.release_date,
            COUNT(c.card_id) AS cards_tracked,
            ROUND(MEDIAN(f.market_price), 2) AS median_market_price,
            ROUND(AVG(f.market_price), 2) AS avg_market_price,
            ROUND(MIN(f.market_price), 2) AS min_market_price,
            ROUND(MAX(f.market_price), 2) AS max_market_price,
            ROUND(AVG(f.price_spread_pct), 2) AS avg_spread_pct,
            SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) AS anomaly_count
        FROM dim_sets s
        JOIN dim_cards c ON s.set_id = c.set_id
        JOIN fact_market_prices f ON c.card_id = f.card_id
        GROUP BY s.set_id, s.set_name, s.series, s.era_category, s.release_date
        ORDER BY median_market_price DESC;
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_top_anomalies(db_path: str = DEFAULT_DB_PATH, limit: int = 30) -> pd.DataFrame:
    """
    Extrae las cartas identificadas como anomalías estadísticas del mercado
    (is_market_anomaly = true), ordenadas por score de anomalía y precio.
    """
    conn = get_connection(db_path)
    try:
        query = f"""
        SELECT 
            c.card_id,
            c.name,
            c.supertype,
            c.subtype,
            c.rarity,
            c.rarity_group,
            s.set_name,
            s.series,
            s.era_category,
            c.artist,
            f.market_price,
            f.low_price,
            f.high_price,
            f.price_spread,
            f.price_spread_pct,
            f.modified_z_score,
            f.price_tier
        FROM fact_market_prices f
        JOIN dim_cards c ON f.card_id = c.card_id
        JOIN dim_sets s ON c.set_id = s.set_id
        WHERE f.is_market_anomaly = TRUE
        ORDER BY f.modified_z_score DESC, f.market_price DESC
        LIMIT {limit};
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_vintage_vs_modern_evolution(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Compara métricas macroeconómicas entre las distintas eras del juego
    (Vintage, Classic/EX, Mid-Era, Modern).
    """
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            s.era_category,
            COUNT(c.card_id) AS total_cards,
            COUNT(DISTINCT s.set_id) AS total_sets,
            ROUND(MEDIAN(f.market_price), 2) AS median_price,
            ROUND(AVG(f.market_price), 2) AS avg_price,
            ROUND(MAX(f.market_price), 2) AS max_price,
            ROUND(AVG(f.price_spread_pct), 2) AS avg_spread_pct,
            SUM(CASE WHEN f.price_tier = 'Grail (>$100)' THEN 1 ELSE 0 END) AS grail_cards_count,
            SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) AS anomaly_cards_count,
            ROUND(100.0 * SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) / COUNT(c.card_id), 2) AS anomaly_percentage
        FROM dim_sets s
        JOIN dim_cards c ON s.set_id = c.set_id
        JOIN fact_market_prices f ON c.card_id = f.card_id
        GROUP BY s.era_category
        ORDER BY median_price DESC;
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_rarity_price_distribution(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Obtiene la distribución cuantitativa de precios por grupo de rareza.
    """
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            c.rarity_group,
            COUNT(c.card_id) AS card_count,
            ROUND(MIN(f.market_price), 2) AS min_price,
            ROUND(QUANTILE_CONT(f.market_price, 0.25), 2) AS p25_price,
            ROUND(MEDIAN(f.market_price), 2) AS median_price,
            ROUND(QUANTILE_CONT(f.market_price, 0.75), 2) AS p75_price,
            ROUND(MAX(f.market_price), 2) AS max_price,
            ROUND(AVG(f.market_price), 2) AS avg_price,
            SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) AS anomalies_count
        FROM dim_cards c
        JOIN fact_market_prices f ON c.card_id = f.card_id
        GROUP BY c.rarity_group
        ORDER BY median_price DESC;
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_top_artists_by_market_value(db_path: str = DEFAULT_DB_PATH, min_cards: int = 5, limit: int = 15) -> pd.DataFrame:
    """
    Identifica a los artistas cuyas cartas alcanzan la mayor valoración mediana en el mercado.
    """
    conn = get_connection(db_path)
    try:
        query = f"""
        SELECT 
            c.artist,
            COUNT(c.card_id) AS total_illustrations,
            ROUND(MEDIAN(f.market_price), 2) AS median_card_price,
            ROUND(AVG(f.market_price), 2) AS avg_card_price,
            ROUND(MAX(f.market_price), 2) AS most_expensive_card_price,
            SUM(CASE WHEN f.price_tier = 'Grail (>$100)' THEN 1 ELSE 0 END) AS grails_count
        FROM dim_cards c
        JOIN fact_market_prices f ON c.card_id = f.card_id
        WHERE c.artist IS NOT NULL AND c.artist != 'Desconocido'
        GROUP BY c.artist
        HAVING COUNT(c.card_id) >= {min_cards}
        ORDER BY median_card_price DESC, avg_card_price DESC
        LIMIT {limit};
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def get_full_cards_dataset(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Retorna la vista completa desnormalizada para filtros interactivos del dashboard.
    """
    conn = get_connection(db_path)
    try:
        query = """
        SELECT 
            c.card_id,
            c.name,
            c.supertype,
            c.subtype,
            c.rarity,
            c.rarity_group,
            c.artist,
            s.set_id,
            s.set_name,
            s.series,
            s.release_date,
            s.era_category,
            f.market_price,
            f.low_price,
            f.high_price,
            f.price_spread,
            f.price_spread_pct,
            f.price_tier,
            f.card_age_years,
            f.modified_z_score,
            f.is_outlier_iqr,
            f.is_outlier_zscore,
            f.is_market_anomaly
        FROM dim_cards c
        JOIN dim_sets s ON c.set_id = s.set_id
        JOIN fact_market_prices f ON c.card_id = f.card_id;
        """
        return conn.execute(query).df()
    finally:
        conn.close()


def execute_custom_query(query: str, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """
    Permite ejecutar consultas SQL personalizadas de solo lectura de forma segura.
    """
    conn = get_connection(db_path)
    try:
        return conn.execute(query).df()
    finally:
        conn.close()


if __name__ == "__main__":
    kpis = get_market_kpis()
    print("\n--- KPIs de Mercado ---")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
