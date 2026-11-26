"""
Dashboard Analítico de Pokémon TCG.
Diseño Editorial e Inteligencia Cuantitativa (Estilo Financial Times / The Economist).
Motor OLAP embebido: DuckDB | Visualización: Plotly + Streamlit.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Importación de módulos locales sin tocar src/ ni la lógica de base de datos
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.analytics import (
    get_market_kpis,
    get_median_price_by_set,
    get_top_anomalies,
    get_vintage_vs_modern_evolution,
    get_rarity_price_distribution,
    get_top_artists_by_market_value,
    get_full_cards_dataset,
    execute_custom_query,
    DEFAULT_DB_PATH
)
from run_pipeline import run_pipeline

# -----------------------------------------------------------------------------
# Configuración de Página de Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pokémon TCG | Market Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Paleta Editorial Restringida (Financial Times / The Economist)
# -----------------------------------------------------------------------------
COLOR_BASE = "#64748B"       # Gris pizarra (datos base / regular)
COLOR_ACCENT = "#8B5CF6"     # Violeta moderno (foco primario / destaques)
COLOR_ANOMALY = "#F59E0B"    # Ámbar editorial (alertas y anomalías estadísticas)
COLOR_BORDER = "#1F2937"     # Bordes sutiles
COLOR_CARD_BG = "#11161F"    # Fondo de tarjetas y paneles

# -----------------------------------------------------------------------------
# Inyección de CSS Editorial Sobrio
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* Ocultar barra de Streamlit y pie de página */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; display: none; }

    /* Reducción de espaciado superior excesivo */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1320px;
    }

    /* Tipografía global sobria */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #E5E7EB;
    }

    /* Encabezado Editorial */
    .editorial-header {
        border-bottom: 1px solid #1F2937;
        padding-bottom: 1.1rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }
    .editorial-title {
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: #F9FAFB;
        margin: 0;
        line-height: 1.2;
    }
    .editorial-subtitle {
        font-size: 0.86rem;
        color: #9CA3AF;
        margin-top: 4px;
        line-height: 1.4;
    }
    .editorial-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        color: #9CA3AF;
        text-align: right;
        letter-spacing: 0.03em;
    }

    /* Tarjetas de Métricas KPI Personalizadas */
    .kpi-card {
        background-color: #11161F;
        border: 1px solid #1F2937;
        border-radius: 4px;
        padding: 14px 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #8B5CF6;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.7rem;
        font-weight: 600;
        color: #F9FAFB;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }
    .kpi-subtext {
        font-size: 11px;
        color: #6B7280;
        margin-top: 5px;
        font-family: 'Inter', sans-serif;
    }

    /* Pestañas Editoriales Limpias */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: transparent;
        border-bottom: 1px solid #1F2937;
        padding: 0;
        margin-bottom: 1.4rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.86rem;
        font-weight: 500;
        color: #9CA3AF;
        padding: 10px 20px;
        background-color: transparent;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        transition: all 0.15s ease;
        letter-spacing: 0.01em;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F3F4F6;
        background-color: transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #8B5CF6 !important;
        border-bottom: 2px solid #8B5CF6 !important;
        font-weight: 600;
    }

    /* Paneles y Contenedores */
    .editorial-panel {
        background-color: #11161F;
        border: 1px solid #1F2937;
        border-radius: 4px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .section-heading {
        font-size: 0.82rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
    }

    /* Tablas y DataFrames */
    .stDataFrame {
        border: 1px solid #1F2937 !important;
        border-radius: 4px !important;
    }

    /* Barra lateral sobria */
    section[data-testid="stSidebar"] {
        background-color: #0E131B !important;
        border-right: 1px solid #1F2937 !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Funciones Auxiliares de Visualización (Plotly Editorial)
# -----------------------------------------------------------------------------
def apply_editorial_theme(fig, height: int = 370):
    """
    Aplica diseño gráfico editorial estricto (Financial Times / The Economist):
    - Sin bordes de caja (showline=False en ambos ejes)
    - Líneas de cuadrícula horizontales muy sutiles (#1F2937)
    - Sin cuadrículas verticales
    - Tipografía sobria (Inter)
    - Fondo 100% transparente
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#9CA3AF", size=11),
        margin=dict(l=35, r=20, t=35, b=35),
        height=height,
        hoverlabel=dict(
            bgcolor="#11161F",
            bordercolor="#38BDF8",
            font=dict(family="Inter, sans-serif", color="#F9FAFB", size=11)
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            linecolor="#1F2937",
            tickfont=dict(color="#9CA3AF", size=11)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1F2937",
            zeroline=False,
            showline=False,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(color="#9CA3AF", size=11)
        )
    )
    return fig


def render_metric_card(label: str, value: str, subtext: str = ""):
    """Genera tarjeta KPI con HTML/CSS personalizado en lugar del diseño plano st.metric."""
    subtext_html = f'<div class="kpi-subtext">{subtext}</div>' if subtext else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {subtext_html}
    </div>
    """


def verify_database_exists() -> bool:
    """Verifica si la base de datos DuckDB existe y contiene registros."""
    if not os.path.exists(DEFAULT_DB_PATH):
        return False
    try:
        kpis = get_market_kpis(DEFAULT_DB_PATH)
        return kpis.get("total_cards", 0) > 0
    except Exception:
        return False


def render_sidebar():
    """Barra lateral sobria de información del sistema y controles de ejecución."""
    with st.sidebar:
        st.markdown("""
        <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #1F2937; margin-bottom: 1.2rem;">
            <div style="font-weight: 700; font-size: 0.95rem; color: #F9FAFB; letter-spacing: -0.01em;">
                POKÉMON TCG ANALYTICS
            </div>
            <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 2px;">
                Motor analítico in-process DuckDB
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Arquitectura del Sistema</div>', unsafe_allow_html=True)
        st.markdown("""
        - Almacén: `DuckDB (Esquema Estrella)`
        - Procesamiento: `Pandas / NumPy`
        - Detección de atípicos: `MAD & IQR`
        - Consultas: `SQL Vectorizado`
        """)

        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Control del Pipeline</div>', unsafe_allow_html=True)
        
        if st.button("Ejecutar Pipeline ETL", use_container_width=True):
            with st.spinner("Ejecutando pipeline ETL..."):
                t0 = time.time()
                run_pipeline()
                elapsed = time.time() - t0
                st.success(f"Pipeline ejecutado en {elapsed:.2f}s")
                st.rerun()

        st.markdown("""
        <div style="font-size: 11px; color: #6B7280; margin-top: 1.5rem; line-height: 1.4;">
            En ausencia de archivos en <code>data/raw/</code>, el módulo de extracción activa automáticamente un generador sintético estocástico de 2.200 cartas.
        </div>
        """, unsafe_allow_html=True)


def main():
    render_sidebar()

    # Encabezado Editorial Superior
    st.markdown("""
    <div class="editorial-header">
        <div>
            <h1 class="editorial-title">Pokémon TCG Market Analytics</h1>
            <div class="editorial-subtitle">
                Análisis cuantitativo de precios, liquidez y detección estadística de anomalías en el mercado secundario.
            </div>
        </div>
        <div class="editorial-meta">
            MOTOR OLAP: DUCKDB<br>
            ACTUALIZACIÓN: EN TIEMPO REAL
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Validar que exista la base de datos DuckDB
    if not verify_database_exists():
        st.warning("No se encontró el almacén analítico DuckDB en 'data/pokemon_market.duckdb'.")
        if st.button("Inicializar Almacén DuckDB y Ejecutar Pipeline", type="primary"):
            with st.spinner("Procesando datos y construyendo esquema estrella..."):
                run_pipeline()
                st.success("Almacén analítico inicializado exitosamente.")
                st.rerun()
        st.stop()

    # Cargar métricas agregadas
    try:
        kpis = get_market_kpis(DEFAULT_DB_PATH)
    except Exception as e:
        st.error(f"Error al conectar con la base de datos DuckDB: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # Estructura Principal en Pestañas Limpias (Sin emojis)
    # -------------------------------------------------------------------------
    tab_overview, tab_sets, tab_anomalies, tab_artists, tab_sql = st.tabs([
        "Resumen Ejecutivo",
        "Expansiones y Series",
        "Detección de Anomalías",
        "Artistas y Rarezas",
        "Consola SQL DuckDB"
    ])

    # =========================================================================
    # TAB 1: RESUMEN EJECUTIVO & KPIS
    # =========================================================================
    with tab_overview:
        # Fila de tarjetas de métricas KPI personalizadas (sin st.metric)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                render_metric_card("Cartas Monitoreadas", f"{int(kpis.get('total_cards', 0)):,}", "Muestra consolidada"),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                render_metric_card("Expansiones", f"{int(kpis.get('total_sets', 0))}", f"{int(kpis.get('total_series', 0))} series históricas"),
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                render_metric_card("Precio Mediano", f"${kpis.get('median_price', 0):.2f}", "USD por unidad"),
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                render_metric_card("Spread Promedio", f"{kpis.get('avg_spread_pct', 0):.1f}%", "Amplitud High-Low"),
                unsafe_allow_html=True
            )
        with col5:
            st.markdown(
                render_metric_card("Anomalías de Mercado", f"{int(kpis.get('total_anomalies', 0))}", f"{kpis.get('anomaly_rate_pct', 0):.1f}% de la muestra"),
                unsafe_allow_html=True
            )

        st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown('<div class="section-heading">Precio Mediano por Era Histórica (USD)</div>', unsafe_allow_html=True)
            df_era = get_vintage_vs_modern_evolution(DEFAULT_DB_PATH)

            # Bar chart editorial: Gris pizarra con destaque en Vintage
            colors_era = [
                COLOR_ACCENT if era == "Vintage" else COLOR_BASE
                for era in df_era["era_category"]
            ]
            fig_era = go.Figure(
                data=[
                    go.Bar(
                        x=df_era["era_category"],
                        y=df_era["median_price"],
                        marker_color=colors_era,
                        text=[f"${v:.2f}" for v in df_era["median_price"]],
                        textposition="outside",
                        textfont=dict(family="IBM Plex Mono", size=11, color="#E5E7EB")
                    )
                ]
            )
            apply_editorial_theme(fig_era, height=340)
            fig_era.update_layout(yaxis_title="Precio Mediano ($)")
            st.plotly_chart(fig_era, use_container_width=True, config={"displayModeBar": False})

        with col_right:
            st.markdown('<div class="section-heading">Top 10 Expansiones por Precio Mediano (USD)</div>', unsafe_allow_html=True)
            df_sets = get_median_price_by_set(DEFAULT_DB_PATH).head(10)

            # Bar horizontal: Gris pizarra con el set líder resaltado en acento
            colors_sets = [COLOR_ACCENT if i == 0 else COLOR_BASE for i in range(len(df_sets))]
            fig_top_sets = go.Figure(
                data=[
                    go.Bar(
                        x=df_sets["median_market_price"],
                        y=df_sets["set_name"],
                        orientation="h",
                        marker_color=colors_sets,
                        text=[f"${v:.2f}" for v in df_sets["median_market_price"]],
                        textposition="outside",
                        textfont=dict(family="IBM Plex Mono", size=10, color="#E5E7EB")
                    )
                ]
            )
            apply_editorial_theme(fig_top_sets, height=340)
            fig_top_sets.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_top_sets, use_container_width=True, config={"displayModeBar": False})

        # Panel de activo líder
        st.markdown(f"""
        <div class="editorial-panel" style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="section-heading" style="margin-bottom: 4px;">Activo con Mayor Cotización Registrada</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #F9FAFB;">
                    {kpis.get('top_card_name')} <span style="color: {COLOR_ACCENT}; font-size: 0.88rem; font-weight: 400;">({kpis.get('top_card_set')})</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div class="kpi-label" style="margin-bottom: 2px;">Precio de Mercado</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: {COLOR_ACCENT}">
                    ${kpis.get('top_card_price', 0):,.2f} <span style="font-size: 0.8rem; color: {COLOR_ACCENT};">USD</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: EXPANSIONES Y SERIES
    # =========================================================================
    with tab_sets:
        st.markdown('<div class="section-heading">Evolución Temporal de Precios por Expansión</div>', unsafe_allow_html=True)
        df_all_sets = get_median_price_by_set(DEFAULT_DB_PATH)

        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            selected_eras = st.multiselect(
                "Filtrar por era:",
                options=df_all_sets["era_category"].unique(),
                default=df_all_sets["era_category"].unique()
            )
        with f_col2:
            search_query = st.text_input("Buscar por nombre o serie:", placeholder="Ej. Base Set, Evolving Skies, Neo...")

        filtered_sets = df_all_sets[df_all_sets["era_category"].isin(selected_eras)]
        if search_query:
            filtered_sets = filtered_sets[
                filtered_sets["set_name"].str.contains(search_query, case=False) |
                filtered_sets["series"].str.contains(search_query, case=False)
            ]

        # Scatter temporal editorial: dos colores (Vintage destacado en acento, resto en gris pizarra)
        scatter_colors = [
            COLOR_ACCENT if era == "Vintage" else COLOR_BASE
            for era in filtered_sets["era_category"]
        ]
        fig_scatter_time = go.Figure(
            data=[
                go.Scatter(
                    x=filtered_sets["release_date"],
                    y=filtered_sets["median_market_price"],
                    mode="markers+text",
                    marker=dict(
                        size=filtered_sets["cards_tracked"] / 8 + 4,
                        color=scatter_colors,
                        opacity=0.85,
                        line=dict(width=1, color="#1F2937")
                    ),
                    text=filtered_sets["set_name"],
                    textposition="top center",
                    textfont=dict(size=9, color="#9CA3AF", family="Inter"),
                    hovertemplate="<b>%{text}</b><br>Lanzamiento: %{x}<br>Precio Mediano: $%{y:.2f}<extra></extra>"
                )
            ]
        )
        apply_editorial_theme(fig_scatter_time, height=400)
        fig_scatter_time.update_layout(yaxis_title="Precio Mediano ($)")
        st.plotly_chart(fig_scatter_time, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-heading">Inventario Detallado de Expansiones</div>', unsafe_allow_html=True)
        display_cols = [
            "set_name", "series", "era_category", "release_date",
            "cards_tracked", "median_market_price", "avg_market_price",
            "max_market_price", "avg_spread_pct", "anomaly_count"
        ]
        rename_dict = {
            "set_name": "Expansión",
            "series": "Serie",
            "era_category": "Era",
            "release_date": "Fecha Lanzamiento",
            "cards_tracked": "Cartas",
            "median_market_price": "Mediana ($)",
            "avg_market_price": "Promedio ($)",
            "max_market_price": "Máximo ($)",
            "avg_spread_pct": "Spread (%)",
            "anomaly_count": "Anomalías"
        }
        st.dataframe(
            filtered_sets[display_cols].rename(columns=rename_dict),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 3: DETECCIÓN DE ANOMALÍAS
    # =========================================================================
    with tab_anomalies:
        st.markdown('<div class="section-heading">Distribución de Precios frente al Z-Score Modificado (MAD)</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="editorial-panel" style="font-size: 0.84rem; color: #9CA3AF; line-height: 1.5;">
            <strong style="color: #F3F4F6;">Metodología de Inferencia Estadística:</strong>
            Las distribuciones de precios en mercados coleccionables son fuertemente asimétricas. El estadístico Z estándar distorsiona la detección al verse afectado por valores astronómicos.
            Este análisis calcula el <strong>Z-Score Modificado (Boris Iglewicz & David Hoaglin, 1993)</strong> basado en la <em>Mediana</em> y la <em>Desviación Absoluta de la Mediana (MAD)</em> estratificado por grupo de rareza.
            Valores con <code>M_i &gt; 3.5</code> concurrentes con el límite superior del <strong>Rango Intercuartílico (IQR)</strong> se clasifican como anomalías de mercado confirmadas.
        </div>
        """, unsafe_allow_html=True)

        df_anomalies = get_top_anomalies(DEFAULT_DB_PATH, limit=120)
        df_full = get_full_cards_dataset(DEFAULT_DB_PATH)

        # Gráfico Editorial de Dos Colores: Normal (Gris pizarra #64748B) vs Anomalías (Ámbar editorial #F59E0B)
        df_normal = df_full[~df_full["is_market_anomaly"]]
        df_anom = df_full[df_full["is_market_anomaly"]]

        fig_outliers = go.Figure()

        # Serie Normal (Gris pizarra)
        fig_outliers.add_trace(
            go.Scatter(
                x=df_normal["market_price"],
                y=df_normal["modified_z_score"],
                mode="markers",
                name="Distribución Regular",
                marker=dict(color=COLOR_BASE, size=6, opacity=0.45),
                hovertemplate="<b>%{text}</b><br>Precio: $%{x:.2f}<br>Z-Score MAD: %{y:.2f}<extra></extra>",
                text=df_normal["name"]
            )
        )

        # Serie Anomalías (Ámbar editorial de acento)
        fig_outliers.add_trace(
            go.Scatter(
                x=df_anom["market_price"],
                y=df_anom["modified_z_score"],
                mode="markers",
                name="Anomalía Confirmada",
                marker=dict(color=COLOR_ANOMALY, size=8, opacity=0.9, line=dict(width=1, color="#11161F")),
                hovertemplate="<b>%{text}</b> [ANOMALÍA]<br>Precio: $%{x:.2f}<br>Z-Score MAD: %{y:.2f}<extra></extra>",
                text=df_anom["name"]
            )
        )

        # Línea de umbral crítico MAD
        fig_outliers.add_hline(
            y=3.5,
            line_dash="dot",
            line_color=COLOR_ANOMALY,
            line_width=1,
            annotation_text="Umbral crítico MAD (3.5)",
            annotation_position="bottom right",
            annotation_font=dict(color=COLOR_ANOMALY, size=10, family="IBM Plex Mono")
        )

        apply_editorial_theme(fig_outliers, height=410)
        fig_outliers.update_layout(
            xaxis_title="Precio de Mercado ($ USD)",
            yaxis_title="Z-Score Modificado (MAD)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color="#9CA3AF")
            )
        )
        st.plotly_chart(fig_outliers, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-heading">Registros Identificados como Anomalías de Mercado</div>', unsafe_allow_html=True)
        col_af1, col_af2 = st.columns([1, 2])
        with col_af1:
            rarity_filter = st.selectbox(
                "Filtrar por grupo de rareza:",
                options=["Todas"] + list(df_anomalies["rarity_group"].unique())
            )

        disp_anomalies = df_anomalies if rarity_filter == "Todas" else df_anomalies[df_anomalies["rarity_group"] == rarity_filter]

        st.dataframe(
            disp_anomalies[[
                "name", "rarity_group", "set_name", "era_category", "market_price",
                "low_price", "high_price", "price_spread_pct", "modified_z_score", "price_tier"
            ]].rename(columns={
                "name": "Carta",
                "rarity_group": "Rareza",
                "set_name": "Expansión",
                "era_category": "Era",
                "market_price": "Precio ($)",
                "low_price": "Mínimo ($)",
                "high_price": "Máximo ($)",
                "price_spread_pct": "Spread (%)",
                "modified_z_score": "Z-Score MAD",
                "price_tier": "Segmento"
            }),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 4: ARTISTAS Y RAREZAS
    # =========================================================================
    with tab_artists:
        col_box, col_art = st.columns([1, 1])

        with col_box:
            st.markdown('<div class="section-heading">Distribución Cuantitativa de Precios por Rareza</div>', unsafe_allow_html=True)
            log_scale = st.checkbox("Escala logarítmica", value=True)
            df_full = get_full_cards_dataset(DEFAULT_DB_PATH)

            # Boxplot editorial: Gris pizarra limpio
            fig_box = go.Figure()
            for r_group in df_full["rarity_group"].unique():
                sub_prices = df_full[df_full["rarity_group"] == r_group]["market_price"]
                fig_box.add_trace(
                    go.Box(
                        y=sub_prices,
                        name=r_group,
                        marker_color=COLOR_ACCENT if "Secret" in r_group or "Ultra" in r_group else COLOR_BASE,
                        line_width=1.2,
                        boxpoints="outliers"
                    )
                )

            apply_editorial_theme(fig_box, height=360)
            fig_box.update_layout(
                showlegend=False,
                yaxis_type="log" if log_scale else "linear",
                yaxis_title="Precio ($ USD)"
            )
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

        with col_art:
            st.markdown('<div class="section-heading">Top Ilustradores por Precio Mediano (USD)</div>', unsafe_allow_html=True)
            df_artists = get_top_artists_by_market_value(DEFAULT_DB_PATH, min_cards=5, limit=10)

            # Bar horizontal: Gris pizarra con líder en color de acento
            colors_art = [COLOR_ACCENT if i == 0 else COLOR_BASE for i in range(len(df_artists))]
            fig_artists = go.Figure(
                data=[
                    go.Bar(
                        x=df_artists["median_card_price"],
                        y=df_artists["artist"],
                        orientation="h",
                        marker_color=colors_art,
                        text=[f"${v:.2f}" for v in df_artists["median_card_price"]],
                        textposition="outside",
                        textfont=dict(family="IBM Plex Mono", size=10, color="#E5E7EB")
                    )
                ]
            )
            apply_editorial_theme(fig_artists, height=360)
            fig_artists.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_artists, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-heading">Composición Jerárquica: Supertipo y Segmento de Valor</div>', unsafe_allow_html=True)
        # Treemap editorial con paleta de dos tonos
        fig_treemap = px.treemap(
            df_full,
            path=["supertype", "rarity_group", "price_tier"],
            values="market_price",
            color="supertype",
            color_discrete_map={
                "Pokémon": COLOR_ACCENT,
                "Trainer": COLOR_BASE,
                "Energy": "#475569"
            }
        )
        fig_treemap.update_traces(
            textfont=dict(family="Inter, sans-serif", size=11),
            marker=dict(line=dict(color="#0B0F17", width=1.5))
        )
        apply_editorial_theme(fig_treemap, height=440)
        st.plotly_chart(fig_treemap, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # TAB 5: DATA WAREHOUSE & SQL CONSOLE
    # =========================================================================
    with tab_sql:
        st.markdown('<div class="section-heading">Consola Analítica Vectorizada (DuckDB In-Process)</div>', unsafe_allow_html=True)

        with st.expander("Esquema relacional estrella (DDL)", expanded=False):
            st.code("""
-- Modelo Dimensional en data/pokemon_market.duckdb
dim_sets           (set_id PK, set_name, series, release_date, era_category, total_cards_count)
dim_cards          (card_id PK, name, supertype, subtype, rarity, rarity_group, artist, set_id)
fact_market_prices (fact_id PK, card_id, market_price, low_price, high_price, price_spread,
                    price_spread_pct, price_tier, card_age_years, modified_z_score,
                    is_outlier_iqr, is_outlier_zscore, is_market_anomaly, load_timestamp)
            """, language="sql")

        predefined_queries = {
            "Top 10 cartas Vintage más cotizadas": """
SELECT c.name, s.set_name, s.era_category, f.market_price, f.modified_z_score
FROM fact_market_prices f
JOIN dim_cards c ON f.card_id = c.card_id
JOIN dim_sets s ON c.set_id = s.set_id
WHERE s.era_category = 'Vintage'
ORDER BY f.market_price DESC
LIMIT 10;
""",
            "Tasa de anomalías y spreads por serie": """
SELECT 
    s.series,
    COUNT(c.card_id) AS total_cartas,
    SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) AS anomalias,
    ROUND(100.0 * SUM(CASE WHEN f.is_market_anomaly THEN 1 ELSE 0 END) / COUNT(c.card_id), 2) AS pct_anomalias,
    ROUND(AVG(f.price_spread_pct), 2) AS spread_promedio
FROM dim_sets s
JOIN dim_cards c ON s.set_id = c.set_id
JOIN fact_market_prices f ON c.card_id = f.card_id
GROUP BY s.series
ORDER BY pct_anomalias DESC;
""",
            "Agregaciones por escalón de precio": """
SELECT 
    price_tier,
    COUNT(*) AS total_cartas,
    ROUND(MIN(market_price), 2) AS min_precio,
    ROUND(MEDIAN(market_price), 2) AS mediana_precio,
    ROUND(MAX(market_price), 2) AS max_precio
FROM fact_market_prices
GROUP BY price_tier
ORDER BY mediana_precio DESC;
"""
        }

        col_q1, col_q2 = st.columns([1, 2])
        with col_q1:
            selected_example = st.selectbox("Consultas predefinidas:", list(predefined_queries.keys()))

        sql_input = st.text_area(
            "Editor SQL (Solo lectura):",
            value=predefined_queries[selected_example],
            height=130
        )

        if st.button("Ejecutar Consulta", type="primary"):
            try:
                t_q = time.time()
                df_sql = execute_custom_query(sql_input, DEFAULT_DB_PATH)
                q_duration = (time.time() - t_q) * 1000.0

                st.success(f"Consulta ejecutada en {q_duration:.2f} ms | {len(df_sql):,} registros retornados")
                st.dataframe(df_sql, use_container_width=True)

                csv_data = df_sql.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Exportar CSV",
                    data=csv_data,
                    file_name="duckdb_query_result.csv",
                    mime="text/csv"
                )
            except Exception as sql_err:
                st.error(f"Error al ejecutar consulta SQL: {sql_err}")


if __name__ == "__main__":
    main()
