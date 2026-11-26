"""
Módulo de Transformación (Capa Silver) para el pipeline ETL de Pokémon TCG.

Integra DuckDB (`read_csv_auto(..., union_by_name = true)`) para la limpieza columnar,
normalización de fechas, casteo a DOUBLE, filtrado de precios inválidos y deduplicación
por carta/snapshot, preservando al 100% todas las reglas de transformación y detección
estadística de anomalías existentes:
1. Limpieza monetaria y formateo numérico robusto (conversión de strings con '$', comas y espacios).
2. Normalización de campos textuales y categorización de rarezas (`rarity_group`).
3. Feature Engineering analítico:
   - Antigüedad de la carta (`card_age_years`).
   - Clasificación cronológica por Era (`era_category`: Vintage, Classic/EX, Mid-Era, Modern).
   - Métricas de liquidez y dispersión: `price_spread` y `price_spread_pct`.
   - Segmentación de mercado: `price_tier` (Budget, Mid-Range, Premium, Grail).
4. Detección estadística avanzada de anomalías y outliers:
   - Rango Intercuartílico (IQR) agrupado por rareza.
   - Z-Score Modificado basado en Mediana y Desviación Absoluta de la Mediana (MAD).
   - Bandera de consenso analítico `is_market_anomaly`.
"""

import os
import uuid
import tempfile
import logging
from typing import Tuple, Optional
from datetime import date

import duckdb
import numpy as np
import pandas as pd

# Configuración del registrador de logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [TRANSFORM]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fecha de referencia para el cálculo de antigüedad
REFERENCE_DATE = pd.to_datetime("2026-11-24")


def clean_currency_series(series: pd.Series) -> pd.Series:
    """
    Convierte una serie con posibles valores de moneda en formato string
    (ej. '$12.50', ' 1,200.00 ', None) a float64 limpio redondeado a 2 decimales.
    """
    if pd.api.types.is_numeric_dtype(series):
        cleaned = series.astype(float)
    else:
        cleaned = (
            series.astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        cleaned = pd.to_numeric(cleaned, errors="coerce")

    cleaned = cleaned.fillna(0.10).clip(lower=0.01)
    return cleaned.round(2)


def assign_era(release_date_series: pd.Series) -> pd.Series:
    """
    Clasifica las fechas de lanzamiento en eras históricas del juego:
    'Vintage', 'Classic/EX', 'Mid-Era', 'Modern'.
    """
    conditions = [
        release_date_series < pd.Timestamp("2003-01-01"),
        (release_date_series >= pd.Timestamp("2003-01-01")) & (release_date_series < pd.Timestamp("2011-01-01")),
        (release_date_series >= pd.Timestamp("2011-01-01")) & (release_date_series < pd.Timestamp("2020-01-01")),
        release_date_series >= pd.Timestamp("2020-01-01"),
    ]
    choices = ["Vintage", "Classic/EX", "Mid-Era", "Modern"]
    return pd.Series(np.select(conditions, choices, default="Modern"), index=release_date_series.index)


def standardize_rarity(rarity_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Limpia el texto de rareza y genera un grupo analítico homogéneo.
    """
    clean_rarity = rarity_series.fillna("Unknown").astype(str).str.strip()

    def map_rarity_group(val: str) -> str:
        low = val.lower()
        if "common" in low and "uncommon" not in low:
            return "Common"
        elif "uncommon" in low:
            return "Uncommon"
        elif any(k in low for k in ["secret", "rainbow", "hyper rare", "special illustration", "illustration rare", "gold"]):
            return "Secret Rare"
        elif any(k in low for k in ["ultra", "vmax", "vstar", " ex", " gx", "radiant", "ace spec", "shiny"]):
            return "Ultra Rare"
        elif "holo" in low or "foil" in low:
            return "Rare Holo"
        elif "promo" in low:
            return "Promo"
        elif "rare" in low:
            return "Rare"
        return "Special / Other"

    rarity_group = clean_rarity.apply(map_rarity_group)
    return clean_rarity, rarity_group


def categorize_price_tier(price_series: pd.Series) -> pd.Series:
    """Segmenta las cartas en escalones de precio de mercado para análisis de coleccionismo."""
    conditions = [
        price_series < 5.0,
        (price_series >= 5.0) & (price_series < 25.0),
        (price_series >= 25.0) & (price_series < 100.0),
        price_series >= 100.0,
    ]
    tiers = ["Budget (<$5)", "Mid-Range ($5-$25)", "Premium ($25-$100)", "Grail (>$100)"]
    return pd.Series(np.select(conditions, tiers, default="Budget (<$5)"), index=price_series.index)


def detect_statistical_anomalies(
    df: pd.DataFrame,
    group_col: str = "rarity_group",
    price_col: str = "market_price",
) -> pd.DataFrame:
    """
    Detecta anomalías de precios agrupando por categoría de rareza utilizando dos métodos:
    1. IQR (Rango Intercuartílico): Q1 - 1.5 * IQR | Q3 + 1.5 * IQR
    2. Z-Score Modificado (Iglewicz & Hoaglin, 1993):
       M_i = (0.6745 * |x_i - Mediana|) / MAD, umbral crítico M_i > 3.5
    """
    logger.info(f"Calculando detección estadística de anomalías agrupada por '{group_col}'...")

    df_result = df.copy()
    df_result["is_outlier_iqr"] = False
    df_result["modified_z_score"] = 0.0
    df_result["is_outlier_zscore"] = False
    df_result["is_market_anomaly"] = False

    groups = df_result.groupby(group_col)

    for _, indices in groups.groups.items():
        subset_prices = df_result.loc[indices, price_col].values

        if len(subset_prices) < 4:
            continue

        # 1. Método IQR
        q1 = np.percentile(subset_prices, 25)
        q3 = np.percentile(subset_prices, 75)
        iqr = q3 - q1
        upper_iqr_limit = q3 + 1.5 * iqr
        lower_iqr_limit = max(0.0, q1 - 1.5 * iqr)

        iqr_mask = (subset_prices > upper_iqr_limit) | (subset_prices < lower_iqr_limit)
        df_result.loc[indices, "is_outlier_iqr"] = iqr_mask

        # 2. Método Z-Score Modificado (MAD)
        median = np.median(subset_prices)
        abs_deviations = np.abs(subset_prices - median)
        mad = np.median(abs_deviations)

        if mad > 1e-4:
            mod_z = 0.6745 * abs_deviations / mad
        else:
            std_dev = np.std(subset_prices)
            if std_dev > 1e-4:
                mod_z = np.abs(subset_prices - median) / std_dev
            else:
                mod_z = np.zeros_like(subset_prices)

        mod_z_rounded = np.round(mod_z, 2)
        zscore_mask = mod_z_rounded > 3.5

        df_result.loc[indices, "modified_z_score"] = mod_z_rounded
        df_result.loc[indices, "is_outlier_zscore"] = zscore_mask

    # Consenso analítico
    df_result["is_market_anomaly"] = (
        (df_result["is_outlier_iqr"] & df_result["is_outlier_zscore"])
        | (df_result["modified_z_score"] > 6.0)
    )

    total_iqr = int(df_result["is_outlier_iqr"].sum())
    total_zscore = int(df_result["is_outlier_zscore"].sum())
    total_anomalies = int(df_result["is_market_anomaly"].sum())

    logger.info(
        f"Detección completada: {total_iqr:,} outliers IQR, {total_zscore:,} outliers Z-Score, "
        f"{total_anomalies:,} anomalías confirmadas ({total_anomalies / max(len(df_result), 1):.1%})."
    )
    return df_result


def build_silver_base_with_duckdb(df: pd.DataFrame, snapshot_date: Optional[str] = None) -> pd.DataFrame:
    """
    Ejecuta la limpieza columnar Silver en DuckDB utilizando `read_csv_auto(..., union_by_name = true)`:
    - Normaliza fechas (`release_date` -> DATE en formato YYYY-MM-DD).
    - Castea precios a `DOUBLE` limpiando caracteres monetarios (`$`, `,`).
    - Filtra precios inválidos (`market_price IS NULL OR market_price <= 0`).
    - Deduplica por carta y snapshot (`QUALIFY ROW_NUMBER() OVER (PARTITION BY card_id, snapshot_date ...)`).
    """
    snap = snapshot_date or date.today().isoformat()
    tmp_csv = os.path.join(tempfile.gettempdir(), f"bronze_stage_{uuid.uuid4().hex}.csv")

    try:
        df.to_csv(tmp_csv, index=False)
        csv_sql_path = tmp_csv.replace("\\", "/")

        conn = duckdb.connect(database=":memory:")
        try:
            query = f"""
            WITH raw_bronze AS (
                SELECT *
                FROM read_csv_auto('{csv_sql_path}', union_by_name = true, all_varchar = true)
            ),
            typed_silver AS (
                SELECT
                    TRIM(CAST(card_id AS VARCHAR)) AS card_id,
                    TRIM(COALESCE(CAST(name AS VARCHAR), 'Desconocido')) AS name,
                    TRIM(COALESCE(NULLIF(CAST(supertype AS VARCHAR), ''), 'Pokémon')) AS supertype,
                    TRIM(COALESCE(NULLIF(CAST(subtype AS VARCHAR), ''), 'Standard')) AS subtype,
                    TRIM(COALESCE(NULLIF(CAST(rarity AS VARCHAR), ''), 'Unknown')) AS rarity,
                    TRIM(COALESCE(NULLIF(CAST(artist AS VARCHAR), ''), 'Desconocido')) AS artist,
                    TRIM(CAST(set_id AS VARCHAR)) AS set_id,
                    TRIM(COALESCE(NULLIF(CAST(set_name AS VARCHAR), ''), 'Desconocido')) AS set_name,
                    TRIM(COALESCE(NULLIF(CAST(series AS VARCHAR), ''), 'Desconocido')) AS series,
                    COALESCE(
                        TRY_CAST(release_date AS DATE),
                        TRY_STRPTIME(TRIM(CAST(release_date AS VARCHAR)), '%m/%d/%Y')::DATE,
                        TRY_STRPTIME(TRIM(CAST(release_date AS VARCHAR)), '%Y-%m-%d')::DATE,
                        DATE '2020-01-01'
                    ) AS release_date_parsed,
                    TRY_CAST(REPLACE(REPLACE(TRIM(CAST(market_price AS VARCHAR)), '$', ''), ',', '') AS DOUBLE) AS raw_market_price,
                    TRY_CAST(REPLACE(REPLACE(TRIM(CAST(low_price AS VARCHAR)), '$', ''), ',', '') AS DOUBLE) AS raw_low_price,
                    TRY_CAST(REPLACE(REPLACE(TRIM(CAST(high_price AS VARCHAR)), '$', ''), ',', '') AS DOUBLE) AS raw_high_price,
                    DATE '{snap}' AS snapshot_date
                FROM raw_bronze
                WHERE card_id IS NOT NULL AND TRIM(CAST(card_id AS VARCHAR)) != ''
            ),
            filtered_silver AS (
                SELECT
                    card_id,
                    name,
                    supertype,
                    subtype,
                    rarity,
                    artist,
                    set_id,
                    set_name,
                    series,
                    STRFTIME(release_date_parsed, '%Y-%m-%d') AS release_date,
                    ROUND(raw_market_price, 2) AS market_price,
                    ROUND(LEAST(COALESCE(NULLIF(raw_low_price, 0.0), raw_market_price * 0.85), raw_market_price), 2) AS low_price,
                    ROUND(GREATEST(COALESCE(NULLIF(raw_high_price, 0.0), raw_market_price * 1.30), raw_market_price), 2) AS high_price,
                    STRFTIME(snapshot_date, '%Y-%m-%d') AS snapshot_date
                FROM typed_silver
                WHERE raw_market_price IS NOT NULL AND raw_market_price > 0.0
            )
            SELECT *
            FROM filtered_silver
            QUALIFY ROW_NUMBER() OVER (PARTITION BY card_id, snapshot_date ORDER BY release_date DESC) = 1;
            """
            df_silver = conn.execute(query).df()
        finally:
            conn.close()

        return df_silver
    finally:
        if os.path.exists(tmp_csv):
            try:
                os.remove(tmp_csv)
            except Exception:
                pass


def transform_pokemon_data(df: pd.DataFrame, snapshot_date: Optional[str] = None) -> pd.DataFrame:
    """
    Orquesta la transformación completa hacia la Capa Silver (`silver_cards`):
    1. Normalización, casteo a DOUBLE, filtrado de precios inválidos y deduplicación vía DuckDB (`read_csv_auto`).
    2. Limpieza monetaria adicional y consistencia `low_price <= market_price <= high_price`.
    3. Normalización de rarezas (`rarity_group`).
    4. Cálculo de antigüedad (`card_age_years`) y era histórica (`era_category`).
    5. Feature Engineering: `price_spread`, `price_spread_pct`, `price_tier`.
    6. Detección estadística estratificada de anomalías (IQR + Z-Score Modificado MAD).
    """
    logger.info(f"Iniciando transformación a Capa Silver ({len(df):,} filas de entrada)...")

    # 1. Limpieza columnar, fechas, casteo DOUBLE y deduplicación en DuckDB
    transformed = build_silver_base_with_duckdb(df, snapshot_date=snapshot_date)
    logger.info(f"Capa base DuckDB Silver construida: {len(transformed):,} filas válidas y deduplicadas.")

    # 2. Garantizar limpieza monetaria y coherencia lógica
    transformed["market_price"] = clean_currency_series(transformed["market_price"])
    transformed["low_price"] = clean_currency_series(transformed["low_price"])
    transformed["high_price"] = clean_currency_series(transformed["high_price"])
    transformed["low_price"] = np.minimum(transformed["low_price"], transformed["market_price"])
    transformed["high_price"] = np.maximum(transformed["high_price"], transformed["market_price"])

    # 3. Normalización de rareza
    clean_rarity, rarity_group = standardize_rarity(transformed["rarity"])
    transformed["rarity"] = clean_rarity
    transformed["rarity_group"] = rarity_group

    # 4. Parsing de fechas y antigüedad
    parsed_dates = pd.to_datetime(transformed["release_date"], errors="coerce").fillna(pd.Timestamp("2020-01-01"))
    transformed["release_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    transformed["card_age_years"] = np.round((REFERENCE_DATE - parsed_dates).dt.days / 365.25, 1)
    transformed["era_category"] = assign_era(parsed_dates)

    # 5. Feature Engineering: Spreads y Tiers
    transformed["price_spread"] = (transformed["high_price"] - transformed["low_price"]).round(2)
    spread_pct = ((transformed["price_spread"] / transformed["market_price"]) * 100.0).round(2)
    transformed["price_spread_pct"] = spread_pct.clip(lower=0.0, upper=500.0)
    transformed["price_tier"] = categorize_price_tier(transformed["market_price"])

    # 6. Detección de anomalías estadísticas
    transformed = detect_statistical_anomalies(transformed, group_col="rarity_group", price_col="market_price")

    logger.info(f"Transformación Silver completada con éxito. Dimensiones finales: {transformed.shape}.")
    return transformed


if __name__ == "__main__":
    from extract import extract_raw_data
    df_raw = extract_raw_data()
    df_clean = transform_pokemon_data(df_raw)
    print("\n--- Vista previa de Capa Silver ---")
    print(df_clean[["card_id", "name", "set_name", "rarity_group", "market_price", "era_category", "is_market_anomaly"]].head(5))
