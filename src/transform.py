"""
Módulo de Transformación para el pipeline ETL de Pokémon TCG.

Este módulo realiza:
1. Limpieza monetaria y formateo numérico robusto (conversión de strings con '$', comas y espacios).
2. Normalización de campos textuales y categorización de rarezas.
3. Feature Engineering analítico:
   - Antigüedad de la carta (card_age_years).
   - Clasificación cronológica por Era (Vintage, Clásica, Media, Moderna).
   - Métricas de liquidez y dispersión: price_spread y price_spread_pct.
   - Segmentación de mercado: price_tier (Budget, Mid-range, Premium, Grail).
4. Detección estadística avanzada de anomalías y outliers:
   - Rango Intercuartílico (IQR) agrupado por rareza.
   - Z-Score Modificado basado en Mediana y Desviación Absoluta de la Mediana (MAD).
   - Bandera de consenso analítico 'is_market_anomaly'.
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd

# Configuración del registrador de logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [TRANSFORM]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Fecha de referencia para el cálculo de antigüedad (Año 2026)
REFERENCE_DATE = pd.to_datetime("2026-11-24")


def clean_currency_series(series: pd.Series) -> pd.Series:
    """
    Convierte una serie con posibles valores de moneda en formato string
    (ej. '$12.50', ' 1,200.00 ', None) a float64 limpio redondeado a 2 decimales.

    Args:
        series: Serie de pandas a limpiar.

    Returns:
        pd.Series de tipo float64 con valores válidos >= 0.01.
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

    # Reemplazar valores nulos o negativos con un valor base mínimo
    cleaned = cleaned.fillna(0.10).clip(lower=0.01)
    return cleaned.round(2)


def assign_era(release_date_series: pd.Series) -> pd.Series:
    """
    Clasifica las fechas de lanzamiento en eras históricas del juego.

    Args:
        release_date_series: Serie datetime.

    Returns:
        pd.Series con categorías: 'Vintage', 'Classic/EX', 'Mid-Era', 'Modern'.
    """
    conditions = [
        release_date_series < pd.Timestamp("2003-01-01"),
        (release_date_series >= pd.Timestamp("2003-01-01")) & (release_date_series < pd.Timestamp("2011-01-01")),
        (release_date_series >= pd.Timestamp("2011-01-01")) & (release_date_series < pd.Timestamp("2020-01-01")),
        release_date_series >= pd.Timestamp("2020-01-01")
    ]
    choices = ["Vintage", "Classic/EX", "Mid-Era", "Modern"]
    return pd.Series(np.select(conditions, choices, default="Modern"), index=release_date_series.index)


def standardize_rarity(rarity_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Limpia el texto de rareza y genera un grupo analítico homogéneo.

    Returns:
        Tupla de (rareza_limpia, grupo_rareza_estandarizado).
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
    """
    Segmenta las cartas en escalones de precio de mercado para análisis de coleccionismo.
    """
    conditions = [
        price_series < 5.0,
        (price_series >= 5.0) & (price_series < 25.0),
        (price_series >= 25.0) & (price_series < 100.0),
        price_series >= 100.0
    ]
    tiers = ["Budget (<$5)", "Mid-Range ($5-$25)", "Premium ($25-$100)", "Grail (>$100)"]
    return pd.Series(np.select(conditions, tiers, default="Budget (<$5)"), index=price_series.index)


def detect_statistical_anomalies(df: pd.DataFrame, group_col: str = "rarity_group", price_col: str = "market_price") -> pd.DataFrame:
    """
    Detecta anomalías de precios agrupando por categoría de rareza utilizando dos métodos:
    1. IQR (Rango Intercuartílico):
       Q1 - 1.5 * IQR | Q3 + 1.5 * IQR
    2. Z-Score Modificado (Boris Iglewicz & David Hoaglin, 1993):
       M_i = (0.6745 * |x_i - Mediana|) / MAD
       Umbral crítico estándar: M_i > 3.5

    Ambos métodos se ejecutan por grupo de rareza para evitar que cartas comunes baratas
    opacen los saltos extremos o que cartas secretas de alto valor se consideren falsos positivos.

    Args:
        df: DataFrame con las cartas transformadas.
        group_col: Columna para agrupación estratificada.
        price_col: Columna de precio a evaluar.

    Returns:
        DataFrame con columnas agregadas:
        ['is_outlier_iqr', 'modified_z_score', 'is_outlier_zscore', 'is_market_anomaly']
    """
    logger.info(f"Calculando detección estadística de anomalías agrupada por '{group_col}'...")

    df_result = df.copy()
    
    # Inicialización de banderas
    df_result["is_outlier_iqr"] = False
    df_result["modified_z_score"] = 0.0
    df_result["is_outlier_zscore"] = False
    df_result["is_market_anomaly"] = False

    groups = df_result.groupby(group_col)

    for group_name, indices in groups.groups.items():
        subset_prices = df_result.loc[indices, price_col].values
        
        if len(subset_prices) < 4:
            # Grupo muy pequeño para estadística robusta
            continue

        # --- 1. Método IQR ---
        q1 = np.percentile(subset_prices, 25)
        q3 = np.percentile(subset_prices, 75)
        iqr = q3 - q1
        upper_iqr_limit = q3 + 1.5 * iqr
        lower_iqr_limit = max(0.0, q1 - 1.5 * iqr)

        iqr_mask = (subset_prices > upper_iqr_limit) | (subset_prices < lower_iqr_limit)
        df_result.loc[indices, "is_outlier_iqr"] = iqr_mask

        # --- 2. Método Z-Score Modificado (MAD) ---
        median = np.median(subset_prices)
        abs_deviations = np.abs(subset_prices - median)
        mad = np.median(abs_deviations)

        if mad > 1e-4:
            mod_z = 0.6745 * abs_deviations / mad
        else:
            # Si el MAD es prácticamente cero (valores homogéneos), usar desviación estándar como alternativa
            std_dev = np.std(subset_prices)
            if std_dev > 1e-4:
                mod_z = np.abs(subset_prices - median) / std_dev
            else:
                mod_z = np.zeros_like(subset_prices)

        mod_z_rounded = np.round(mod_z, 2)
        zscore_mask = mod_z_rounded > 3.5

        df_result.loc[indices, "modified_z_score"] = mod_z_rounded
        df_result.loc[indices, "is_outlier_zscore"] = zscore_mask

    # Consenso: anomalía robusta cuando coincide el IQR con Z-Score Modificado > 3.5
    # o cuando el Z-Score Modificado es extremo (> 6.0)
    df_result["is_market_anomaly"] = (
        (df_result["is_outlier_iqr"] & df_result["is_outlier_zscore"]) |
        (df_result["modified_z_score"] > 6.0)
    )

    total_iqr = df_result["is_outlier_iqr"].sum()
    total_zscore = df_result["is_outlier_zscore"].sum()
    total_anomalies = df_result["is_market_anomaly"].sum()

    logger.info(
        f"Detección completada: {total_iqr:,} outliers IQR, {total_zscore:,} outliers Z-Score, "
        f"{total_anomalies:,} anomalías de mercado confirmadas ({total_anomalies/len(df_result):.1%})."
    )
    return df_result


def transform_pokemon_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función principal que orquesta todas las transformaciones del pipeline:
    - Limpieza monetaria de market_price, low_price y high_price.
    - Limpieza de cadenas de texto y deduplicación.
    - Normalización de rarezas y supertipos.
    - Parsing de fechas y cálculo de antigüedad (card_age_years).
    - Asignación de eras históricas (era_category).
    - Cálculo de spreads de precios (price_spread, price_spread_pct).
    - Segmentación por price_tier.
    - Detección estadística multivariada de anomalías.

    Args:
        df: DataFrame crudo extraído.

    Returns:
        pd.DataFrame completamente transformado, enriquecido y tipado.
    """
    logger.info(f"Iniciando transformación de datos ({len(df):,} filas)...")
    transformed = df.copy()

    # 1. Limpieza de identificadores y textos
    transformed["card_id"] = transformed["card_id"].astype(str).str.strip()
    transformed["name"] = transformed["name"].astype(str).str.strip()
    transformed["set_id"] = transformed["set_id"].astype(str).str.strip()
    transformed["set_name"] = transformed["set_name"].astype(str).str.strip()
    transformed["series"] = transformed["series"].astype(str).str.strip()
    transformed["artist"] = transformed["artist"].fillna("Desconocido").astype(str).str.strip()
    transformed["supertype"] = transformed["supertype"].fillna("Pokémon").astype(str).str.strip()
    transformed["subtype"] = transformed["subtype"].fillna("Standard").astype(str).str.strip()

    # 2. Limpieza monetaria
    transformed["market_price"] = clean_currency_series(transformed["market_price"])
    transformed["low_price"] = clean_currency_series(transformed["low_price"])
    transformed["high_price"] = clean_currency_series(transformed["high_price"])

    # Asegurar coherencia lógica entre precios (low <= market <= high)
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
    # Spread relativo en porcentaje sobre el precio de mercado
    spread_pct = ((transformed["price_spread"] / transformed["market_price"]) * 100.0).round(2)
    transformed["price_spread_pct"] = spread_pct.clip(lower=0.0, upper=500.0)
    transformed["price_tier"] = categorize_price_tier(transformed["market_price"])

    # 6. Detección de anomalías estadísticas
    transformed = detect_statistical_anomalies(transformed, group_col="rarity_group", price_col="market_price")

    logger.info(f"Transformación completada con éxito. Dimensiones finales: {transformed.shape}.")
    return transformed


if __name__ == "__main__":
    from extract import extract_raw_data
    df_raw = extract_raw_data()
    df_clean = transform_pokemon_data(df_raw)
    print("\n--- Vista previa de datos transformados ---")
    print(df_clean[["name", "set_name", "rarity_group", "market_price", "era_category", "modified_z_score", "is_market_anomaly"]].head(5))
