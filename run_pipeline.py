"""
Orquestador principal del Pipeline ETL de Pokémon TCG (Arquitectura Medallion).

Ejecuta el flujo completo de ingeniería de datos:
1. EXTRACCIÓN (Bronze): Validación y lectura de CSVs en `data/raw/` (o fallback sintético).
2. TRANSFORMACIÓN (Silver - DuckDB): Limpieza columnar con `read_csv_auto`, normalización de
   fechas/rarezas, features de mercado y detección estadística de anomalías (IQR + Z-Score MAD).
3. CARGA (Silver Parquet & Gold DuckDB): Persistencia de `silver_cards` y `gold_set_summary`
   en `data/warehouse/pokemon_tcg.duckdb` + exportación a `data/processed/cards_clean.parquet`.
4. HEALTH CHECK & VALIDACIÓN: Consultas analíticas inmediatas para verificar integridad.
"""

import sys
import os
import time
import argparse
import logging

# Añadir directorio actual al path para importación de paquetes
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.extract import extract_raw_data, generate_synthetic_pokemon_data
from src.transform import transform_pokemon_data
from src.load import load_to_duckdb, DEFAULT_DB_PATH, DEFAULT_PARQUET_PATH
from src.analytics import get_market_kpis, get_top_anomalies

# Configuración de Logging profesional para consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ETL_ORCHESTRATOR")


def print_banner():
    banner = r"""
================================================================================
     POKÉMON TCG - ETL & ANALYTICS PIPELINE (MEDALLION: BRONZE -> SILVER -> GOLD)
================================================================================
    Stack: Python 3.12 | DuckDB (OLAP) | Parquet (ZSTD) | Pandas | Plotly
================================================================================
"""
    print(banner)


def run_pipeline(
    raw_dir: str = "data/raw",
    db_path: str = DEFAULT_DB_PATH,
    parquet_path: str = DEFAULT_PARQUET_PATH,
    force_synthetic: bool = False,
):
    """
    Ejecuta el pipeline ETL completo registrando métricas de tiempo y volumen.
    """
    total_start_time = time.time()
    print_banner()

    logger.info(">>> INICIANDO PIPELINE DE DATOS POKÉMON TCG <<<")
    logger.info(f"Capa Bronze (datos crudos):   {os.path.abspath(raw_dir)}")
    logger.info(f"Capa Silver (Parquet):        {os.path.abspath(parquet_path)}")
    logger.info(f"Capa Gold (DuckDB Warehouse): {os.path.abspath(db_path)}")

    # -------------------------------------------------------------
    # FASE 1: EXTRACCIÓN (BRONZE)
    # -------------------------------------------------------------
    stage1_start = time.time()
    logger.info("=" * 60)
    logger.info("[FASE 1/3] EXTRACCIÓN DE DATOS (BRONZE)")
    logger.info("=" * 60)

    if force_synthetic:
        logger.info("Opción --force-synthetic activada: Generando dataset sintético directamente...")
        df_raw = generate_synthetic_pokemon_data(n_cards=2200)
    else:
        df_raw = extract_raw_data(raw_dir=raw_dir)

    stage1_duration = time.time() - stage1_start
    logger.info(f"[FASE 1 COMPLETADA] {len(df_raw):,} registros extraídos en {stage1_duration:.2f}s.")

    # -------------------------------------------------------------
    # FASE 2: TRANSFORMACIÓN (SILVER - DUCKDB)
    # -------------------------------------------------------------
    stage2_start = time.time()
    logger.info("=" * 60)
    logger.info("[FASE 2/3] TRANSFORMACIÓN COLUMNAR DUCKDB & ANOMALÍAS (SILVER)")
    logger.info("=" * 60)

    df_transformed = transform_pokemon_data(df_raw)

    stage2_duration = time.time() - stage2_start
    logger.info(f"[FASE 2 COMPLETADA] {len(df_transformed):,} registros Silver procesados en {stage2_duration:.2f}s.")

    # -------------------------------------------------------------
    # FASE 3: CARGA EN DUCKDB & PARQUET (SILVER + GOLD)
    # -------------------------------------------------------------
    stage3_start = time.time()
    logger.info("=" * 60)
    logger.info("[FASE 3/3] CARGA EN DUCKDB (SILVER/GOLD) Y EXPORTACIÓN PARQUET")
    logger.info("=" * 60)

    load_summary = load_to_duckdb(
        df_transformed,
        db_path=db_path,
        parquet_path=parquet_path,
    )

    stage3_duration = time.time() - stage3_start
    logger.info(
        f"[FASE 3 COMPLETADA] Carga finalizada en {stage3_duration:.2f}s | "
        f"silver_cards={load_summary['silver_cards']:,}, gold_set_summary={load_summary['gold_set_summary']:,}"
    )

    # -------------------------------------------------------------
    # FASE 4: VALIDACIÓN ANALÍTICA POST-CARGA (HEALTH CHECK)
    # -------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("[POST-LOAD CHECK] RESUMEN ANALÍTICO DEL WAREHOUSE")
    logger.info("=" * 60)

    kpis = get_market_kpis(db_path=db_path)
    print("\n" + "-" * 60)
    print("  MÉTRICAS CLAVE DEL DATA WAREHOUSE (DUCKDB + PARQUET)")
    print("-" * 60)
    print(f"  * Filas en silver_cards:             {load_summary['silver_cards']:,}")
    print(f"  * Expansiones en gold_set_summary:   {load_summary['gold_set_summary']:,}")
    print(f"  * Total de series históricas:        {kpis.get('total_series', 0)}")
    print(f"  * Precio mediano de mercado:         ${kpis.get('median_price', 0):.2f}")
    print(f"  * Precio promedio de mercado:        ${kpis.get('avg_price', 0):.2f}")
    print(f"  * Spread promedio de mercado:        {kpis.get('avg_spread_pct', 0):.2f}%")
    print(f"  * Carta más valiosa registrada:      {kpis.get('top_card_name', '')} (${kpis.get('top_card_price', 0):.2f}) en {kpis.get('top_card_set', '')}")
    print(f"  * Anomalías estadísticas detectadas: {kpis.get('total_anomalies', 0):,} ({kpis.get('anomaly_rate_pct', 0):.2f}% del mercado)")
    print("-" * 60 + "\n")

    top_anomalies = get_top_anomalies(db_path=db_path, limit=5)
    print("Top 5 anomalías estadísticas más destacadas (Z-Score Modificado):")
    for idx, row in top_anomalies.iterrows():
        print(
            f"  [{idx + 1}] {row['name']} ({row['rarity_group']}) en {row['set_name']} | "
            f"Precio: ${row['market_price']:.2f} | Z-Score: {row['modified_z_score']:.2f}"
        )

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 80)
    logger.info(f"PIPELINE FINALIZADO CON ÉXITO en {total_duration:.2f} segundos.")
    print("================================================================================\n")
    return load_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutor del Pipeline ETL de Pokémon TCG")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Ruta al directorio de datos crudos (Bronze)")
    parser.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH, help="Ruta al warehouse DuckDB (Gold)")
    parser.add_argument("--parquet-path", type=str, default=DEFAULT_PARQUET_PATH, help="Ruta al archivo Parquet (Silver)")
    parser.add_argument("--force-synthetic", action="store_true", help="Forzar la generación de datos sintéticos")

    args = parser.parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        db_path=args.db_path,
        parquet_path=args.parquet_path,
        force_synthetic=args.force_synthetic,
    )
