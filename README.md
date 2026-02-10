# 🏪 RetailIQ Analytics Dashboard

**Plataforma integral de análisis de ventas retail con ETL, EDA e IA Generativa**

---

## 📋 Descripción del Problema

Las tiendas retail generan miles de transacciones diarias que contienen información valiosa sobre el comportamiento de compra, preferencias de producto, estacionalidad y eficacia de descuentos. Sin embargo, estos datos frecuentemente contienen errores, valores faltantes e inconsistencias que dificultan el análisis.

**RetailIQ Analytics** es una aplicación web que integra el ciclo completo de la Ciencia de Datos (ETL → EDA → IA Generativa) para analizar un dataset de ventas retail con imperfecciones reales (valores ERROR, UNKNOWN, nulos, outliers). La plataforma responde tres preguntas estratégicas:

1. **¿Qué categorías generan mayor ingreso y cuáles dependen más de descuentos para vender?** — Para optimizar la estrategia de pricing y promociones.
2. **¿Existe estacionalidad en las ventas y cómo varía la demanda a lo largo del tiempo?** — Para planificar inventario y campañas.
3. **¿Cómo difiere el comportamiento de compra entre canales (Online vs In‑store) y métodos de pago?** — Para optimizar la estrategia omnicanal.

---

## 🏗️ Arquitectura Técnica

```
┌──────────────────────────────────────────────────────┐
│                  STREAMLIT FRONTEND                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │   ETL    │  │   EDA    │  │   IA Insights     │  │
│  │  Module  │  │  Module  │  │   (Groq API)      │  │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
│       │              │                 │              │
│  ┌────┴──────────────┴─────────────────┴──────────┐  │
│  │       Pandas · NumPy · Plotly · data_cleaning   │  │
│  └────────────────────┬───────────────────────────┘  │
│                       │                              │
│  ┌────────────────────┴───────────────────────────┐  │
│  │         CSV / JSON / URL Data Sources           │  │
│  └─────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Instalación y Ejecución Local

### Prerrequisitos
- Python 3.10+
- Cuenta gratuita en [Groq](https://console.groq.com/keys) para la API Key

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/retailiq-analytics.git
cd retailiq-analytics

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`. Desde la interfaz, suba el dataset `sales_retail-store.csv` (incluido en `datasets/`) vía upload, o cualquier otro CSV/JSON/URL.

---

## 🌐 Link al Despliegue

> **Streamlit Cloud:** [https://retailiq-analytics.streamlit.app](https://retailiq-analytics.streamlit.app)


---

## 📁 Estructura del Repositorio

```
retailiq-analytics/
├── .streamlit/                    # Configuración de tema (colores EAFIT)
│   └── config.toml
├── datasets/                      # Dataset original de Kaggle
│   └── sales_retail-store.csv
├── overview/                    # Enunciado del proyecto y manual de usuario
│   └── final-project_statement_data-science.pdf
│   └── user-manual.pdf
├── app.py                         # Código principal de la aplicación
├── data_cleaning.py               # Módulo de limpieza y feature engineering
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Esta documentación
├── manual_usuario.pdf             # Guía PDF para el usuario final
└── .gitignore                     # Archivos excluidos de Git
```

---

## 📊 Dataset

**Fuente:** [Retail Store Sales: Dirty for Data Cleaning](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning) — Kaggle

| Columna | Tipo | Descripción |
|---------|------|-------------|
| Transaction ID | ID | Identificador único de transacción |
| Customer ID | Categórica | Identificador del cliente |
| Category | Categórica nominal | Categoría del producto (8 categorías) |
| Item | Categórica nominal | Nombre específico del producto |
| Quantity | Numérica discreta | Unidades compradas |
| Price Per Unit | Numérica continua | Precio unitario |
| Total Spent | Numérica continua | Gasto total de la transacción |
| Payment Method | Categórica nominal | Método de pago |
| Location | Categórica nominal | Canal de venta (Online/In-store) |
| Transaction Date | Temporal | Fecha de la transacción |
| Discount Applied | Booleana | Si se aplicó descuento |

**Imperfecciones intencionadas:** valores `ERROR`, `UNKNOWN`, `NONE`, campos numéricos como texto, nulos distribuidos, fechas inconsistentes.

---

## 🔧 Módulos de la Aplicación

### Módulo 1: ETL (Ingesta y Procesamiento)
- Carga dinámica desde **CSV, JSON o URL**.
- **Auditoría de calidad** con Health Score (Completitud × Unicidad × Validez).
- Limpieza interactiva: eliminación de duplicados, reemplazo de tokens inválidos, conversión de tipos.
- **Recálculo inteligente:** recupera `Total Spent`, `Price Per Unit` o `Quantity` cuando faltan, usando la relación `Total = Qty × Price`.
- Tratamiento de outliers (IQR×3, capeo).
- Feature Engineering: `Ticket_Promedio`, `Mes`, `Día_Semana`, `Trimestre`, `Rango_Gasto`, `Es_FinDeSemana`.

### Módulo 2: EDA (Visualización Dinámica)
- Filtros globales (sidebar): fechas, categorías, ubicación, método de pago, slider de gasto.
- **Tab Univariado:** Histogramas y boxplots interactivos con color por categoría (Plotly).
- **Tab Bivariado:** Heatmap de correlaciones, scatter con trendlines OLS, evolución temporal.
- **Tab Reporte Estratégico:** Gráficos que responden las 3 preguntas de negocio con tablas de resumen.

### Módulo 3: IA Insights (Groq)
- Conexión a Groq API (Llama-3.3 70B / Llama-3.1 8B / Mixtral 8×7B).
- Envío automático de `df.describe()` + agregados por categoría y canal + contexto de negocio.
- Generación de tendencias, riesgos, oportunidades y segmentación en lenguaje natural.
- Descarga de insights en TXT.

---

## 🎓 Créditos

**Autores:**
- **Gia Mariana Calle Higuita**
- **José Santiago Molano Perdomo**
- **Juan José Restrepo Higuita**

**Docente:** Jorge Iván Padilla-Buriticá
**Curso:** SI6001 - Fundamentos en Ciencia de Datos

**Fuente de datos:** [Kaggle — Retail Store Sales: Dirty for Data Cleaning](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning) por Ahmed Mohamed.

---

## 📄 Licencia

Proyecto académico — Universidad EAFIT, 2026.
