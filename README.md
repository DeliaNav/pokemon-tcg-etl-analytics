# Pokémon TCG ETL & Market Analytics

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP%20Engine-FFF000.svg?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Parquet](https://img.shields.io/badge/Apache%20Parquet-Columnar%20Storage-50ABF1.svg)](https://parquet.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Data%20Viz-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)

Pipeline ETL modular de ingeniería de datos y plataforma analítica interactiva sobre el mercado secundario de **Pokémon TCG**, diseñado bajo una arquitectura **Medallion (Bronze → Silver → Gold)** con **DuckDB** y almacenamiento columnar **Apache Parquet**.

---

## Arquitectura Medallion

```mermaid
flowchart TD
    subgraph Bronze ["1. Capa Bronze (src/extract.py)"]
        Raw["CSVs en data/raw/\n(pokemon-tcg-data-master + tcg_prices_cache)"] --> Validate["Validación de CSVs\n+ Interfaz ExternalPriceProvider"]
        Validate --> Fallback["Fallback Sintético\n(2.200 cartas si no hay CSVs)"]
    end

    subgraph Silver ["2. Capa Silver (src/transform.py -> Parquet)"]
        Validate & Fallback --> DuckSilver["DuckDB read_csv_auto(..., union_by_name=true)\n- Fechas normalizadas\n- Precios DOUBLE & filtro > 0\n- Deduplicación por card_id/snapshot"]
        DuckSilver --> Features["Feature Engineering & Anomalías:\n- era_category, card_age_years\n- price_spread, price_tier\n- IQR + Z-Score Modificado (MAD)"]
        Features --> Parquet["data/processed/cards_clean.parquet\n(Compresión ZSTD)"]
    end

    subgraph Gold ["3. Capa Gold (src/load.py -> DuckDB Warehouse)"]
        Features --> Warehouse["data/warehouse/pokemon_tcg.duckdb"]
        Warehouse --> SilverTable["silver_cards"]
        Warehouse --> GoldTable["gold_set_summary\n(total_cards, avg, median, max, price_stddev)"]
    end

    subgraph Analytics ["4. Explotación & BI (src/analytics.py + dashboard.py)"]
        Parquet & GoldTable --> Dashboard["Dashboard Streamlit + Plotly\n(@st.cache_data + Consola SQL DuckDB)"]
    end
```

---

## Stack Técnico

- **Lenguaje:** Python 3.12+
- **Motor OLAP:** DuckDB (`read_csv_auto`, SQL vectorizado, exportación `COPY ... TO ... (FORMAT PARQUET)`)
- **Almacenamiento Columnar:** Apache Parquet (`pyarrow` / ZSTD)
- **Procesamiento & Estadística:** Pandas, NumPy, SciPy (IQR y Z-Score Modificado / MAD estratificado por rareza)
- **Visualización:** Streamlit + Plotly (diseño editorial oscuro con `@st.cache_data`)
- **Calidad & CI/CD:** Pytest + GitHub Actions (`.github/workflows/etl_pipeline.yml`)

---

## Estructura del Proyecto

```text
pokemon-tcg-etl-analytics/
├── .github/workflows/
│   └── etl_pipeline.yml          # CI: ejecuta run_pipeline.py y pytest
├── .streamlit/
│   └── config.toml               # Tema oscuro editorial y toolbar minimal
├── data/
│   ├── raw/                      # Capa Bronze (CSVs crudos / caché de precios)
│   ├── processed/
│   │   └── cards_clean.parquet   # Capa Silver columnar exportada desde DuckDB
│   └── warehouse/
│       └── pokemon_tcg.duckdb    # Data Warehouse DuckDB (silver_cards + gold_set_summary)
├── src/
│   ├── __init__.py
│   ├── extract.py                # Validación Bronze, proveedor externo desacoplado y fallback sintético
│   ├── transform.py              # Limpieza SQL DuckDB, feature engineering y anomalías (IQR + MAD)
│   ├── load.py                   # Persistencia DuckDB (Silver + Gold) y exportación Parquet
│   └── analytics.py              # Consultas OLAP sobre DuckDB y Parquet
├── tests/
│   └── test_pipeline.py          # Tests unitarios y de integración (esquemas, Parquet, idempotencia)
├── run_pipeline.py               # Orquestador CLI (Extract -> Transform -> Load)
├── dashboard.py                  # Dashboard analítico interactivo en Streamlit + Plotly
├── requirements.txt
└── README.md
```

---

## Ejecución en Local

### 1. Crear entorno virtual e instalar dependencias
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Ejecutar el Pipeline ETL completo
```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```
Opciones CLI disponibles:
- `--force-synthetic`: fuerza el dataset sintético (útil en entornos sin CSVs).
- `--raw-dir`: ruta alternativa de datos crudos (`data/raw` por defecto).
- `--db-path`: ruta del warehouse DuckDB (`data/warehouse/pokemon_tcg.duckdb` por defecto).
- `--parquet-path`: ruta del Parquet Silver (`data/processed/cards_clean.parquet` por defecto).

### 3. Ejecutar las pruebas automatizadas
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### 4. Lanzar el Dashboard Analítico
```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```
Abrir en el navegador `http://localhost:8501`.
