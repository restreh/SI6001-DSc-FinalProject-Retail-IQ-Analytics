"""
=============================================================================
 RetailIQ Analytics Dashboard
 Proyecto Final — Ciencia de Datos e IA Generativa
 Autores: Gia Mariana Calle Higuita
          José Santiago Molano Perdomo
          Juan José Restrepo Higuita
 Universidad EAFIT · 2025
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import io
import json

from data_cleaning import (
    calculate_health_score,
    clean_retail_data,
    create_derived_features,
    generate_cleaning_report,
    detect_outliers_iqr,
)

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="RetailIQ Analytics",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        font-size: 2.5rem; font-weight: 700; color: #1E3A5F;
        text-align: center; margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa; border-radius: 10px; padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1E3A5F; }
    .kpi-label { font-size: 0.85rem; color: #666; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6; border-radius: 4px; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A5F !important; color: white !important;
    }

    .warning-box {
        background-color: #fff3cd; border: 1px solid #ffc107;
        border-radius: 5px; padding: 1rem; margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda; border: 1px solid #28a745;
        border-radius: 5px; padding: 1rem; margin: 1rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #1E3A5F;
        padding: 1.2rem 1.5rem; border-radius: 0 10px 10px 0; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

@st.cache_data
def load_csv_file(file):
    try:
        return pd.read_csv(file, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin-1", low_memory=False)


@st.cache_data
def load_json_file(file):
    data = json.load(file)
    return pd.json_normalize(data) if isinstance(data, list) else pd.DataFrame([data])


def format_currency(val):
    return f"${val:,.2f}"


def format_percentage(val):
    return f"{val:.1f}%"


def create_health_comparison_chart(report):
    """Gráfico de barras comparando métricas antes vs después."""
    metrics = ["health_score", "completitud", "unicidad", "validez"]
    labels = ["Health Score", "Completitud", "Unicidad", "Validez"]
    before = [report["metricas_antes"].get(m, 0) for m in metrics]
    after = [report["metricas_despues"].get(m, 0) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Antes", x=labels, y=before,
                         marker_color="#ef4444", opacity=0.7))
    fig.add_trace(go.Bar(name="Después", x=labels, y=after,
                         marker_color="#22c55e", opacity=0.85))
    fig.update_layout(
        barmode="group", template="plotly_white",
        title="Métricas de Calidad: Antes vs Después",
        yaxis_title="Porcentaje (%)", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_nullity_heatmap(null_dict, title="Nulidad por Columna"):
    """Heatmap de nulidad residual por columna."""
    cols = list(null_dict.keys())
    vals = list(null_dict.values())
    fig = px.bar(
        x=vals, y=cols, orientation="h",
        labels={"x": "% Nulos", "y": "Columna"},
        title=title, template="plotly_white",
        color=vals, color_continuous_scale="OrRd",
    )
    fig.update_layout(height=max(300, len(cols) * 32), showlegend=False)
    return fig


# =============================================================================
# SIDEBAR — NAVEGACIÓN Y FILTROS
# =============================================================================

def render_sidebar():
    st.sidebar.markdown("## 🏪 RetailIQ Analytics")
    st.sidebar.markdown("---")

    modulo = st.sidebar.radio(
        "Navegación",
        ["🏠 Inicio",
         "🔬 Módulo 1: ETL",
         "📊 Módulo 2: EDA",
         "🤖 Módulo 3: IA Insights"],
        index=0,
    )

    filters = {}

    if "df_final" in st.session_state and st.session_state.df_final is not None:
        df = st.session_state.df_final
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Resumen")
        st.sidebar.info(
            f"**Registros:** {len(df):,}\n\n"
            f"**Columnas:** {df.shape[1]}\n\n"
            f"**Periodo:** {df['Transaction Date'].min().strftime('%Y-%m-%d') if 'Transaction Date' in df.columns and df['Transaction Date'].notna().any() else 'N/A'} a "
            f"{df['Transaction Date'].max().strftime('%Y-%m-%d') if 'Transaction Date' in df.columns and df['Transaction Date'].notna().any() else 'N/A'}"
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Filtros Globales")

        # Filtro de fechas
        if "Transaction Date" in df.columns and df["Transaction Date"].notna().any():
            col1, col2 = st.sidebar.columns(2)
            min_d = df["Transaction Date"].min().date()
            max_d = df["Transaction Date"].max().date()
            with col1:
                filters["fecha_inicio"] = st.date_input("Desde", value=min_d, min_value=min_d, max_value=max_d)
            with col2:
                filters["fecha_fin"] = st.date_input("Hasta", value=max_d, min_value=min_d, max_value=max_d)

        # Filtro de categoría
        if "Category" in df.columns:
            cats = sorted(df["Category"].dropna().unique().tolist())
            filters["categorias"] = st.sidebar.multiselect("Categorías", options=cats, default=[])

        # Filtro de ubicación
        if "Location" in df.columns:
            locs = sorted(df["Location"].dropna().unique().tolist())
            filters["ubicaciones"] = st.sidebar.multiselect("Ubicación", options=locs, default=[])

        # Filtro de método de pago
        if "Payment Method" in df.columns:
            pays = sorted(df["Payment Method"].dropna().unique().tolist())
            filters["pagos"] = st.sidebar.multiselect("Método de Pago", options=pays, default=[])

        # Slider de gasto
        if "Total Spent" in df.columns:
            mn, mx = float(df["Total Spent"].min()), float(df["Total Spent"].max())
            if mn < mx:
                filters["gasto_rango"] = st.sidebar.slider(
                    "Rango Total Spent", mn, mx, (mn, mx)
                )

        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 Refrescar", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align:center; font-size:0.7rem; color:#94a3b8;'>"
        "© 2025 Gia Calle · Santiago Molano · Juan José Restrepo<br>"
        "Universidad EAFIT</div>",
        unsafe_allow_html=True,
    )

    return modulo, filters


def apply_filters(df, filters):
    """Aplica los filtros globales al DataFrame."""
    dff = df.copy()
    if filters.get("fecha_inicio") and filters.get("fecha_fin") and "Transaction Date" in dff.columns:
        dff = dff[
            (dff["Transaction Date"] >= pd.Timestamp(filters["fecha_inicio"])) &
            (dff["Transaction Date"] <= pd.Timestamp(filters["fecha_fin"]))
        ]
    if filters.get("categorias"):
        dff = dff[dff["Category"].isin(filters["categorias"])]
    if filters.get("ubicaciones"):
        dff = dff[dff["Location"].isin(filters["ubicaciones"])]
    if filters.get("pagos"):
        dff = dff[dff["Payment Method"].isin(filters["pagos"])]
    if filters.get("gasto_rango") and "Total Spent" in dff.columns:
        dff = dff[(dff["Total Spent"] >= filters["gasto_rango"][0]) &
                  (dff["Total Spent"] <= filters["gasto_rango"][1])]
    return dff


# =============================================================================
# TAB: INICIO
# =============================================================================

def render_inicio():
    st.markdown('<p class="main-header">🏪 RetailIQ Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Dashboard de Análisis de Ventas Retail · ETL · EDA · IA Generativa</p>', unsafe_allow_html=True)

    st.markdown("### 🎯 Preguntas Estratégicas de Negocio")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="metric-card">'
            '<p class="kpi-label"><b>Pregunta 1</b></p>'
            '<p>¿Qué categorías de productos generan mayor ingreso y cuáles dependen más de los descuentos para vender?</p>'
            '</div>', unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="metric-card">'
            '<p class="kpi-label"><b>Pregunta 2</b></p>'
            '<p>¿Existe estacionalidad en las ventas y cómo varía la demanda a lo largo del tiempo?</p>'
            '</div>', unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="metric-card">'
            '<p class="kpi-label"><b>Pregunta 3</b></p>'
            '<p>¿Cómo difiere el comportamiento de compra entre canales (Online vs In‑store) y métodos de pago?</p>'
            '</div>', unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🚀 Cómo usar esta plataforma")
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**🔬 Paso 1: ETL**")
        st.markdown("Cargue su dataset, visualice la auditoría de calidad, limpie datos y genere variables derivadas.")
    with cb:
        st.markdown("**📊 Paso 2: EDA**")
        st.markdown("Explore distribuciones, correlaciones, tendencias temporales y responda las preguntas de negocio.")
    with cc:
        st.markdown("**🤖 Paso 3: IA**")
        st.markdown("Genere insights automáticos con Groq + Llama-3 sobre sus datos filtrados.")


# =============================================================================
# TAB: ETL
# =============================================================================

def render_etl():
    st.markdown('<p class="main-header">🔬 Ingesta y Procesamiento (ETL)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Carga · Auditoría · Limpieza · Feature Engineering</p>', unsafe_allow_html=True)

    # ── INGESTA ──────────────────────────────────────────────────────
    st.markdown("### 📂 1. Ingesta de Datos")
    source = st.radio("Fuente de datos:", ["📁 Subir archivo (CSV/JSON)", "🌐 URL directa"], horizontal=True)

    df_loaded = None
    if source == "📁 Subir archivo (CSV/JSON)":
        uploaded = st.file_uploader("Seleccione un archivo", type=["csv", "json"], key="etl_upload")
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df_loaded = load_csv_file(uploaded)
                else:
                    df_loaded = load_json_file(uploaded)
                st.success(f"✅ **{uploaded.name}** cargado — {df_loaded.shape[0]:,} filas × {df_loaded.shape[1]} columnas")
            except Exception as e:
                st.error(f"❌ Error al leer el archivo: {e}")
    else:
        url = st.text_input("URL del dataset (CSV o JSON):")
        if url:
            try:
                df_loaded = pd.read_json(url) if url.endswith(".json") else pd.read_csv(url, low_memory=False)
                st.success(f"✅ Datos cargados desde URL — {df_loaded.shape[0]:,} filas × {df_loaded.shape[1]} columnas")
            except Exception as e:
                st.error(f"❌ Error al leer la URL: {e}")

    if df_loaded is not None:
        st.session_state.df_raw = df_loaded.copy()

    if "df_raw" not in st.session_state or st.session_state.df_raw is None:
        st.info("👆 Cargue un dataset para comenzar.")
        return

    df_raw = st.session_state.df_raw

    with st.expander("👁️ Vista previa de datos crudos", expanded=False):
        st.dataframe(df_raw.head(100), use_container_width=True, height=300)

    # ── AUDITORÍA ANTES ──────────────────────────────────────────────
    st.markdown("### 🩺 2. Auditoría de Calidad (Datos Crudos)")
    health_before = calculate_health_score(df_raw, "Retail Sales")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Registros", f"{health_before['total_registros']:,}")
    col2.metric("Columnas", f"{health_before['total_columnas']}")
    col3.metric("Health Score", f"{health_before['health_score']:.1f}%")
    col4.metric("Celdas Nulas", f"{health_before['celdas_nulas']:,}")
    col5.metric("Duplicados", f"{health_before['registros_duplicados']:,}")

    with st.expander("📋 Nulidad por columna (datos crudos)", expanded=False):
        fig_null_before = create_nullity_heatmap(health_before["nulidad_por_columna"], "% Nulidad — Datos Crudos")
        st.plotly_chart(fig_null_before, use_container_width=True)

    # ── OPCIONES DE LIMPIEZA ─────────────────────────────────────────
    st.markdown("### 🧹 3. Limpieza Interactiva")
    c_left, c_right = st.columns(2)
    with c_left:
        remove_dups = st.checkbox("🗑️ Eliminar duplicados", value=True)
        fix_invalid = st.checkbox("🔧 Reemplazar ERROR/UNKNOWN/NONE → NaN", value=True)
    with c_right:
        impute_method = st.selectbox("📐 Imputación numérica", ["Mediana", "Media", "Cero"])
        treat_outliers = st.checkbox("📊 Tratar outliers (IQR×3, capeo)", value=True)

    # ── EJECUTAR LIMPIEZA ────────────────────────────────────────────
    if st.button("⚡ Ejecutar Limpieza y Feature Engineering", type="primary", use_container_width=True):
        with st.spinner("Procesando..."):
            # Limpiar
            df_clean, cleaning_log = clean_retail_data(
                df_raw,
                remove_dups=remove_dups,
                impute_method=impute_method,
                fix_invalid=fix_invalid,
                treat_outliers=treat_outliers,
            )

            # Health después
            health_after = calculate_health_score(df_clean, "Retail Sales")

            # Feature engineering
            df_final = create_derived_features(df_clean)

            # Guardar en session
            st.session_state.df_clean = df_clean
            st.session_state.df_final = df_final
            st.session_state.health_before = health_before
            st.session_state.health_after = health_after
            st.session_state.cleaning_log = cleaning_log
            st.session_state.report = generate_cleaning_report(health_before, health_after, cleaning_log)
            st.session_state.etl_done = True

        st.success(f"🎉 ETL completado — **{df_final.shape[0]:,}** filas × **{df_final.shape[1]}** columnas")
        st.balloons()

    # ── MOSTRAR RESULTADOS SI YA SE EJECUTÓ ──────────────────────────
    if st.session_state.get("etl_done"):
        report = st.session_state.report

        st.markdown("---")
        st.markdown("### 📈 4. Resultados de Limpieza")

        # Health Score comparison
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(
            "Health Score",
            f"{report['metricas_despues']['health_score']:.1f}%",
            delta=f"+{report['mejora_health_score']:.1f}%",
        )
        col_b.metric(
            "Completitud",
            f"{report['metricas_despues']['completitud']:.1f}%",
        )
        col_c.metric(
            "Registros Finales",
            f"{report['metricas_despues']['registros']:,}",
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_health = create_health_comparison_chart(report)
            st.plotly_chart(fig_health, use_container_width=True)
        with col2:
            fig_null_after = create_nullity_heatmap(
                report["nulidad_despues"], "% Nulidad — Después de Limpieza"
            )
            st.plotly_chart(fig_null_after, use_container_width=True)

        # Acciones realizadas
        with st.expander("📋 Acciones de Limpieza Realizadas", expanded=True):
            for accion in report["acciones_realizadas"]:
                st.markdown(f"- {accion}")

        # Imputaciones
        if report.get("imputaciones"):
            with st.expander("🔧 Decisiones de Imputación"):
                for col_name, info in report["imputaciones"].items():
                    st.markdown(
                        f"**{col_name}:** {info['metodo']} → `{info['valor_imputado']}` "
                        f"({info['valores_afectados']} valores) — {info['justificacion']}"
                    )

        # Outliers
        if report.get("outliers_detectados"):
            with st.expander("⚠️ Outliers Detectados y Tratados"):
                for col_name, info in report["outliers_detectados"].items():
                    st.warning(f"**{col_name}:** {info['cantidad']} outliers — {info['accion']}")

        # Feature engineering summary
        df_final = st.session_state.df_final
        new_cols = [c for c in df_final.columns if c not in st.session_state.df_raw.columns]
        if new_cols:
            with st.expander("🧪 Variables Derivadas Creadas"):
                st.markdown(f"**Nuevas columnas ({len(new_cols)}):** {', '.join(new_cols)}")
                st.dataframe(df_final[new_cols].head(20), use_container_width=True)

        # Descargar reporte
        st.markdown("---")
        report_lines = ["REPORTE DE AUDITORÍA DE CALIDAD - RetailIQ Analytics",
                        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 60, ""]
        report_lines.append(f"ANTES: Health Score={report['metricas_antes']['health_score']}%, "
                            f"Registros={report['metricas_antes']['registros']}, "
                            f"Nulos={report['metricas_antes']['celdas_nulas']}, "
                            f"Duplicados={report['metricas_antes']['duplicados']}")
        report_lines.append(f"DESPUÉS: Health Score={report['metricas_despues']['health_score']}%, "
                            f"Registros={report['metricas_despues']['registros']}, "
                            f"Nulos={report['metricas_despues']['celdas_nulas']}, "
                            f"Duplicados={report['metricas_despues']['duplicados']}")
        report_lines.append(f"\nMEJORA: +{report['mejora_health_score']}%\n")
        report_lines.append("ACCIONES:")
        for a in report["acciones_realizadas"]:
            report_lines.append(f"  - {a}")

        st.download_button(
            "📥 Descargar Reporte de Limpieza (TXT)",
            data="\n".join(report_lines),
            file_name=f"reporte_limpieza_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )


# =============================================================================
# TAB: EDA
# =============================================================================

def render_eda(df_filtered):
    st.markdown('<p class="main-header">📊 Visualización Dinámica (EDA)</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis univariado, bivariado y reporte estratégico</p>', unsafe_allow_html=True)

    df = df_filtered
    st.markdown(f"**📋 Registros filtrados:** {len(df):,}")

    # KPIs rápidos
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transacciones", f"{len(df):,}")
    if "Total Spent" in df.columns:
        c2.metric("Ingresos Totales", format_currency(df["Total Spent"].sum()))
        c3.metric("Ticket Promedio", format_currency(df["Total Spent"].mean()))
    if "Quantity" in df.columns:
        c4.metric("Unidades Vendidas", f"{df['Quantity'].sum():,.0f}")
    if "Discount Applied" in df.columns:
        pct_disc = (df["Discount Applied"] == True).mean() * 100
        c5.metric("% con Descuento", format_percentage(pct_disc))

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Análisis Univariado", "🔗 Análisis Bivariado", "📝 Reporte Estratégico"])

    # ── TAB 1: UNIVARIADO ────────────────────────────────────────────
    with tab1:
        st.markdown("### Distribuciones Numéricas")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        col_l, col_r = st.columns(2)
        with col_l:
            num_choice = st.selectbox("Variable numérica:", numeric_cols, key="eda_num")
            chart_type = st.radio("Tipo:", ["Histograma", "Boxplot"], horizontal=True, key="eda_type")
        with col_r:
            color_by = st.selectbox("Colorear por:", ["(Ninguno)"] + cat_cols, key="eda_color")

        color_col = None if color_by == "(Ninguno)" else color_by

        if chart_type == "Histograma":
            fig = px.histogram(df, x=num_choice, color=color_col, nbins=50,
                               template="plotly_white", title=f"Distribución de {num_choice}",
                               color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.box(df, y=num_choice, color=color_col,
                         template="plotly_white", title=f"Boxplot de {num_choice}",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Distribuciones Categóricas")
        if cat_cols:
            cat_choice = st.selectbox("Variable categórica:", cat_cols, key="eda_cat")
            counts = df[cat_choice].value_counts().head(20).reset_index()
            counts.columns = [cat_choice, "count"]
            fig_cat = px.bar(counts, x=cat_choice, y="count",
                             template="plotly_white", title=f"Frecuencia de {cat_choice}",
                             color="count", color_continuous_scale="Blues")
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)

    # ── TAB 2: BIVARIADO ─────────────────────────────────────────────
    with tab2:
        st.markdown("### Mapa de Correlaciones")
        numeric_cols_corr = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_selected = st.multiselect(
            "Columnas para correlación:", numeric_cols_corr,
            default=numeric_cols_corr[:8] if len(numeric_cols_corr) >= 8 else numeric_cols_corr,
            key="eda_corr",
        )
        if len(corr_selected) >= 2:
            corr_matrix = df[corr_selected].corr()
            fig_heat = px.imshow(corr_matrix, text_auto=".2f",
                                 color_continuous_scale="RdBu_r",
                                 template="plotly_white", title="Heatmap de Correlación",
                                 aspect="auto")
            fig_heat.update_layout(height=550)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.markdown("### Scatter — Relación entre variables")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            x_var = st.selectbox("Eje X:", numeric_cols_corr, key="eda_sx")
        with col_s2:
            y_var = st.selectbox("Eje Y:", numeric_cols_corr,
                                 index=min(1, len(numeric_cols_corr) - 1), key="eda_sy")
        with col_s3:
            sc_color = st.selectbox("Color:", ["(Ninguno)"] + cat_cols, key="eda_scolor")

        sc_col = None if sc_color == "(Ninguno)" else sc_color
        sample = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df
        fig_sc = px.scatter(sample, x=x_var, y=y_var, color=sc_col, opacity=0.5,
                            template="plotly_white", title=f"{y_var} vs {x_var}",
                            color_discrete_sequence=px.colors.qualitative.Set2, trendline="ols")
        fig_sc.update_layout(height=480)
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📅 Evolución Temporal")
        if "Transaction Date" in df.columns and df["Transaction Date"].notna().any():
            agg_var = st.selectbox("Variable a agregar:", numeric_cols_corr, key="eda_tagg")
            agg_func = st.selectbox("Función:", ["sum", "mean", "count", "median"], key="eda_tfunc")
            df_time = df.dropna(subset=["Transaction Date"]).set_index("Transaction Date")
            df_time = df_time.resample("M")[agg_var].agg(agg_func).reset_index()
            df_time.columns = ["Fecha", agg_var]
            fig_time = px.area(df_time, x="Fecha", y=agg_var, template="plotly_white",
                               title=f"Evolución mensual de {agg_var} ({agg_func})",
                               color_discrete_sequence=["#1E3A5F"])
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No hay columna de fecha disponible para análisis temporal.")

    # ── TAB 3: REPORTE ESTRATÉGICO ───────────────────────────────────
    with tab3:
        render_strategic_report(df)


def render_strategic_report(df):
    """Gráficos que responden las 3 preguntas de negocio."""

    # ══════════════════════════════════════════════════════════════════
    # PREGUNTA 1: Categorías e impacto de descuentos
    # ══════════════════════════════════════════════════════════════════
    st.markdown("### 💰 Pregunta 1: Ingresos por Categoría e Impacto de Descuentos")
    st.markdown(
        '> *¿Qué categorías generan mayor ingreso y cuáles dependen más de los descuentos para vender?*'
    )

    if "Category" in df.columns and "Total Spent" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            rev_cat = df.groupby("Category", observed=True)["Total Spent"].sum().sort_values(ascending=False).reset_index()
            fig_rev = px.bar(rev_cat, x="Category", y="Total Spent",
                             template="plotly_white", title="Ingresos Totales por Categoría",
                             color="Total Spent", color_continuous_scale="Blues")
            fig_rev.update_layout(height=420)
            st.plotly_chart(fig_rev, use_container_width=True)

        with col2:
            if "Discount Applied" in df.columns:
                disc_cat = df.groupby(["Category", "Discount Applied"], observed=True)["Total Spent"].sum().reset_index()
                disc_cat["Discount Applied"] = disc_cat["Discount Applied"].map({True: "Con Descuento", False: "Sin Descuento"})
                fig_disc = px.bar(disc_cat, x="Category", y="Total Spent", color="Discount Applied",
                                  barmode="group", template="plotly_white",
                                  title="Ingresos: Con vs Sin Descuento",
                                  color_discrete_map={"Con Descuento": "#f59e0b", "Sin Descuento": "#1E3A5F"})
                fig_disc.update_layout(height=420)
                st.plotly_chart(fig_disc, use_container_width=True)

        # Tabla: % de transacciones con descuento por categoría
        if "Discount Applied" in df.columns:
            st.markdown("#### 📋 Dependencia de Descuentos por Categoría")
            dep_disc = df.groupby("Category", observed=True).agg(
                Total_Transacciones=("Transaction ID", "count"),
                Transacciones_Descuento=("Discount Applied", "sum"),
                Ingreso_Total=("Total Spent", "sum"),
                Ticket_Promedio=("Total Spent", "mean"),
            ).reset_index()
            dep_disc["% con Descuento"] = (dep_disc["Transacciones_Descuento"] / dep_disc["Total_Transacciones"] * 100).round(1)
            dep_disc["Ingreso_Total"] = dep_disc["Ingreso_Total"].round(2)
            dep_disc["Ticket_Promedio"] = dep_disc["Ticket_Promedio"].round(2)
            dep_disc = dep_disc.sort_values("% con Descuento", ascending=False)
            st.dataframe(dep_disc, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # PREGUNTA 2: Estacionalidad
    # ══════════════════════════════════════════════════════════════════
    st.markdown("### 📅 Pregunta 2: Estacionalidad de las Ventas")
    st.markdown(
        '> *¿Existe estacionalidad en las ventas y cómo varía la demanda a lo largo del tiempo?*'
    )

    if "Transaction Date" in df.columns and df["Transaction Date"].notna().any():
        col1, col2 = st.columns(2)

        with col1:
            if "Nombre_Mes" in df.columns and "Mes" in df.columns:
                monthly = df.groupby(["Mes", "Nombre_Mes"], observed=True)["Total Spent"].sum().reset_index()
                monthly = monthly.sort_values("Mes")
                fig_month = px.bar(monthly, x="Nombre_Mes", y="Total Spent",
                                   template="plotly_white", title="Ingresos por Mes",
                                   color="Total Spent", color_continuous_scale="Teal")
                fig_month.update_layout(height=420)
                st.plotly_chart(fig_month, use_container_width=True)
            else:
                df_m = df.copy()
                df_m["month"] = df_m["Transaction Date"].dt.month
                monthly = df_m.groupby("month")["Total Spent"].sum().reset_index()
                fig_month = px.bar(monthly, x="month", y="Total Spent",
                                   template="plotly_white", title="Ingresos por Mes")
                st.plotly_chart(fig_month, use_container_width=True)

        with col2:
            if "Dia_Semana" in df.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily = df.groupby("Dia_Semana", observed=True)["Total Spent"].mean().reindex(day_order).reset_index()
                daily.columns = ["Día", "Gasto Promedio"]
                fig_day = px.bar(daily, x="Día", y="Gasto Promedio",
                                 template="plotly_white", title="Gasto Promedio por Día de Semana",
                                 color="Gasto Promedio", color_continuous_scale="Purp")
                fig_day.update_layout(height=420)
                st.plotly_chart(fig_day, use_container_width=True)

        # Evolución temporal con categorías
        if "Category" in df.columns and "Trimestre" in df.columns and "Anio" in df.columns:
            df_ts = df.copy()
            df_ts["Periodo"] = df_ts["Anio"].astype(str) + "-Q" + df_ts["Trimestre"].astype(str)
            ts_cat = df_ts.groupby(["Periodo", "Category"], observed=True)["Total Spent"].sum().reset_index()
            fig_ts = px.line(ts_cat, x="Periodo", y="Total Spent", color="Category",
                             template="plotly_white", title="Evolución Trimestral por Categoría",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig_ts.update_layout(height=450)
            st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Se requiere la columna 'Transaction Date' para el análisis de estacionalidad.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # PREGUNTA 3: Canales y métodos de pago
    # ══════════════════════════════════════════════════════════════════
    st.markdown("### 🏬 Pregunta 3: Comportamiento por Canal y Método de Pago")
    st.markdown(
        '> *¿Cómo difiere el comportamiento de compra entre canales (Online vs In‑store) y métodos de pago?*'
    )

    if "Location" in df.columns and "Total Spent" in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            loc_agg = df.groupby("Location", observed=True).agg(
                Transacciones=("Transaction ID", "count"),
                Ingreso_Total=("Total Spent", "sum"),
                Ticket_Promedio=("Total Spent", "mean"),
            ).reset_index()
            fig_loc = px.bar(loc_agg, x="Location", y="Ingreso_Total",
                             text="Transacciones", template="plotly_white",
                             title="Ingresos por Canal (Ubicación)",
                             color="Location", color_discrete_sequence=["#1E3A5F", "#0ea5e9", "#6366f1"])
            fig_loc.update_layout(height=420)
            st.plotly_chart(fig_loc, use_container_width=True)

        with col2:
            if "Payment Method" in df.columns:
                pay_agg = df.groupby("Payment Method", observed=True).agg(
                    Transacciones=("Transaction ID", "count"),
                    Ticket_Promedio=("Total Spent", "mean"),
                ).reset_index()
                fig_pay = px.bar(pay_agg, x="Payment Method", y="Transacciones",
                                 text="Ticket_Promedio", template="plotly_white",
                                 title="Transacciones y Ticket por Método de Pago",
                                 color="Ticket_Promedio", color_continuous_scale="Teal")
                fig_pay.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
                fig_pay.update_layout(height=420)
                st.plotly_chart(fig_pay, use_container_width=True)

        # Comparativa: Canal × Método de pago
        if "Payment Method" in df.columns:
            st.markdown("#### 📊 Matriz: Canal × Método de Pago")
            cross = df.groupby(["Location", "Payment Method"], observed=True)["Total Spent"].mean().reset_index()
            cross_pivot = cross.pivot(index="Location", columns="Payment Method", values="Total Spent").fillna(0)
            fig_cross = px.imshow(cross_pivot, text_auto="$.0f",
                                  color_continuous_scale="Blues", template="plotly_white",
                                  title="Ticket Promedio: Canal × Método de Pago", aspect="auto")
            fig_cross.update_layout(height=350)
            st.plotly_chart(fig_cross, use_container_width=True)

    # Resumen por Rango de Gasto
    if "Rango_Gasto" in df.columns:
        st.markdown("---")
        st.markdown("#### 📋 Resumen por Segmento de Gasto")
        agg_cols = [c for c in ["Total Spent", "Quantity", "Ticket_Promedio"] if c in df.columns]
        if agg_cols:
            summary = df.groupby("Rango_Gasto", observed=True)[agg_cols].agg(["mean", "sum", "count"]).round(2)
            summary.columns = ["_".join(c) for c in summary.columns]
            st.dataframe(summary, use_container_width=True)


# =============================================================================
# TAB: IA INSIGHTS
# =============================================================================

def render_insights(df_filtered):
    st.markdown('<p class="main-header">🤖 Insights con IA Generativa</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análisis automatizado con Groq · Llama-3 / Mixtral</p>', unsafe_allow_html=True)

    df = df_filtered

    # Config
    st.markdown("### 🔑 Configuración de API")
    api_key = st.text_input(
        "API Key de Groq",
        type="password",
        help="Obtenga su clave gratuita en https://console.groq.com/keys",
        value=os.environ.get("GROQ_API_KEY", ""),
    )

    model = st.selectbox("Modelo LLM:", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ])

    if not api_key:
        st.warning("⚠️ Ingrese su API Key de Groq para generar insights con IA")
        st.info("**¿No tienes una API Key?**\n\n"
                "1. Ve a [console.groq.com](https://console.groq.com/)\n"
                "2. Crea una cuenta gratuita\n"
                "3. Genera una API Key")
        return

    # Resumen de datos
    st.markdown("---")
    st.markdown("### 📊 Datos Filtrados Actuales")
    c1, c2, c3 = st.columns(3)
    c1.metric("Transacciones", f"{len(df):,}")
    if "Total Spent" in df.columns:
        c2.metric("Ingresos", format_currency(df["Total Spent"].sum()))
        c3.metric("Ticket Promedio", format_currency(df["Total Spent"].mean()))

    with st.expander("Ver df.describe()"):
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")
    custom_q = st.text_area(
        "📝 Contexto adicional o pregunta específica (opcional):",
        placeholder="Ej: ¿Qué categorías deberían recibir más inversión en marketing?",
        height=80,
    )

    # Generar
    if st.button("🚀 Generar Recomendaciones Estratégicas", type="primary", use_container_width=True):
        desc_str = df.describe().round(2).to_string()

        # Conteos extra
        extra = ""
        if "Category" in df.columns:
            extra += f"\nTop categorías por ingreso:\n{df.groupby('Category')['Total Spent'].sum().sort_values(ascending=False).head(8).to_string()}\n"
        if "Location" in df.columns:
            extra += f"\nVentas por canal:\n{df.groupby('Location')['Total Spent'].sum().to_string()}\n"
        if "Discount Applied" in df.columns:
            pct = (df["Discount Applied"] == True).mean() * 100
            extra += f"\n% transacciones con descuento: {pct:.1f}%\n"

        questions = """
Preguntas estratégicas:
1. ¿Qué categorías generan mayor ingreso y cuáles dependen más de descuentos?
2. ¿Existe estacionalidad en las ventas y cómo varía la demanda?
3. ¿Cómo difiere el comportamiento de compra entre canales y métodos de pago?
"""
        if custom_q:
            questions += f"\nPregunta adicional del usuario: {custom_q}"

        prompt = f"""Eres un consultor senior de retail analytics. Analiza los siguientes datos de una tienda retail y genera recomendaciones estratégicas en español.

--- RESUMEN ESTADÍSTICO ---
{desc_str}

--- DATOS ADICIONALES ---
Registros: {len(df):,}
{extra}

--- CONTEXTO ---
{questions}

Genera un análisis que incluya:
1. **Tendencias Clave**: 3 patrones principales en los datos.
2. **Riesgos Detectados**: Alertas tempranas y problemas potenciales.
3. **Oportunidades de Negocio**: 3 recomendaciones accionables con evidencia.
4. **Segmentación Sugerida**: Propuesta de segmentos de clientes o productos.

Sé conciso, usa datos concretos del resumen y estructura tu respuesta con encabezados claros."""

        with st.spinner("🧠 Analizando datos con IA..."):
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Eres un consultor senior de retail analytics. Respondes siempre en español."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=3000,
                )
                ai_text = response.choices[0].message.content

                st.markdown("### 💡 Recomendaciones Estratégicas")
                st.markdown("---")
                st.markdown(ai_text)
                st.markdown("---")

                st.download_button(
                    "📥 Descargar Recomendaciones (TXT)",
                    data=f"RECOMENDACIONES ESTRATÉGICAS - RetailIQ Analytics\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{ai_text}",
                    file_name=f"recomendaciones_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
            except ImportError:
                st.error("❌ La librería `groq` no está instalada. Ejecute: `pip install groq`")
            except Exception as e:
                st.error(f"❌ Error al conectar con Groq: {e}")
                st.info("Verifique que su API Key sea válida y tenga créditos disponibles.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    modulo, filters = render_sidebar()

    if modulo == "🏠 Inicio":
        render_inicio()

    elif modulo == "🔬 Módulo 1: ETL":
        render_etl()

    elif modulo == "📊 Módulo 2: EDA":
        if "df_final" not in st.session_state or st.session_state.df_final is None:
            st.warning("⚠️ Primero cargue y procese los datos en el **Módulo 1: ETL**.")
            return
        df_filtered = apply_filters(st.session_state.df_final, filters)
        render_eda(df_filtered)

    elif modulo == "🤖 Módulo 3: IA Insights":
        if "df_final" not in st.session_state or st.session_state.df_final is None:
            st.warning("⚠️ Primero cargue y procese los datos en el **Módulo 1: ETL**.")
            return
        df_filtered = apply_filters(st.session_state.df_final, filters)
        render_insights(df_filtered)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#666; font-size:0.8rem;'>"
        "RetailIQ Analytics | Desarrollado por Gia Mariana Calle Higuita · "
        "José Santiago Molano Perdomo · Juan José Restrepo Higuita | "
        "Universidad EAFIT · 2026</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
