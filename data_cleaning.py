"""
Módulo de Limpieza y Curación de Datos
RetailIQ Analytics - Dashboard de Ventas Retail
Autores: Gia Calle · Santiago Molano · Juan José Restrepo

Funciones para:
- Auditoría de calidad (Health Score)
- Limpieza, normalización e imputación
- Detección y tratamiento de outliers
- Feature engineering
- Generación de reportes de limpieza
"""

import pandas as pd
import numpy as np
from datetime import datetime


# =============================================================================
# MÉTRICAS DE CALIDAD
# =============================================================================

def calculate_health_score(df, dataset_name="Dataset"):
    """
    Calcula un Health Score ponderado para el dataset.
    Completitud 40% · Unicidad 30% · Validez 30%.
    """
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    completitud = ((total_cells - null_cells) / total_cells) * 100 if total_cells > 0 else 0
    unicidad = ((df.shape[0] - duplicates) / df.shape[0]) * 100 if df.shape[0] > 0 else 0

    # Validez: penalizar por valores ERROR/UNKNOWN/None en columnas object
    invalid_count = 0
    for col in df.select_dtypes(include=["object"]).columns:
        invalid_count += df[col].astype(str).str.strip().str.upper().isin(
            ["ERROR", "UNKNOWN", "NONE", "N/A", ""]
        ).sum()
    total_object_cells = df.select_dtypes(include=["object"]).shape[0] * df.select_dtypes(include=["object"]).shape[1]
    validez = ((total_object_cells - invalid_count) / total_object_cells) * 100 if total_object_cells > 0 else 100

    health_score = (completitud * 0.4) + (unicidad * 0.3) + (validez * 0.3)

    null_by_column = df.isnull().sum()
    null_pct_by_column = (null_by_column / len(df) * 100).round(2)

    return {
        "dataset_name": dataset_name,
        "total_registros": df.shape[0],
        "total_columnas": df.shape[1],
        "total_celdas": total_cells,
        "celdas_nulas": int(null_cells),
        "registros_duplicados": int(duplicates),
        "completitud_pct": round(completitud, 2),
        "unicidad_pct": round(unicidad, 2),
        "validez_pct": round(validez, 2),
        "health_score": round(health_score, 2),
        "nulidad_por_columna": null_pct_by_column.to_dict(),
        "columnas_con_nulos": null_by_column[null_by_column > 0].to_dict(),
    }


def detect_outliers_iqr(series, multiplier=1.5):
    """Detecta outliers con IQR. Retorna (mask, lower, upper)."""
    if not np.issubdtype(series.dtype, np.number):
        return pd.Series([False] * len(series), index=series.index), None, None
    clean = series.dropna()
    if len(clean) == 0:
        return pd.Series([False] * len(series), index=series.index), None, None
    Q1 = clean.quantile(0.25)
    Q3 = clean.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - (multiplier * IQR)
    upper = Q3 + (multiplier * IQR)
    mask = (series < lower) | (series > upper)
    return mask, lower, upper


# =============================================================================
# LIMPIEZA PRINCIPAL
# =============================================================================

def clean_retail_data(df, remove_dups=True, impute_method="Mediana",
                      fix_invalid=True, treat_outliers=True):
    """
    Pipeline de limpieza completo para el dataset Retail Store Sales.

    Parámetros:
        df: DataFrame crudo
        remove_dups: eliminar filas duplicadas
        impute_method: 'Media', 'Mediana' o 'Cero' para nulos numéricos
        fix_invalid: reemplazar ERROR/UNKNOWN/NONE por NaN y luego imputar
        treat_outliers: capear outliers en Quantity, Price Per Unit, Total Spent

    Retorna:
        (df_clean, cleaning_log)
    """
    df_clean = df.copy()
    log = {
        "registros_originales": len(df),
        "acciones": [],
        "imputaciones": {},
        "outliers_detectados": {},
        "outliers_dataframes": {},
    }

    # ── 1. Normalizar nombres de columna (strip espacios) ────────────
    df_clean.columns = df_clean.columns.str.strip()
    log["acciones"].append("Nombres de columna normalizados (espacios eliminados)")

    # ── 2. Reemplazar valores inválidos → NaN ────────────────────────
    if fix_invalid:
        invalid_tokens = ["ERROR", "UNKNOWN", "NONE", "N/A", ""]
        replaced_total = 0
        for col in df_clean.columns:
            mask = df_clean[col].astype(str).str.strip().str.upper().isin(invalid_tokens)
            n = mask.sum()
            if n > 0:
                df_clean.loc[mask, col] = np.nan
                replaced_total += n
        if replaced_total > 0:
            log["acciones"].append(
                f"Reemplazados {replaced_total:,} valores inválidos (ERROR/UNKNOWN/NONE) → NaN"
            )

    # ── 3. Eliminar duplicados ───────────────────────────────────────
    if remove_dups:
        dups = df_clean.duplicated().sum()
        if dups > 0:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            log["acciones"].append(f"Eliminadas {dups:,} filas duplicadas")

    # ── 4. Convertir tipos de dato ───────────────────────────────────
    # Numéricas
    for col in ["Quantity", "Price Per Unit", "Total Spent"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    log["acciones"].append("Columnas numéricas convertidas (Quantity, Price Per Unit, Total Spent)")

    # Fecha
    if "Transaction Date" in df_clean.columns:
        df_clean["Transaction Date"] = pd.to_datetime(
            df_clean["Transaction Date"], errors="coerce"
        )
        invalid_dates = df_clean["Transaction Date"].isna().sum()
        if invalid_dates > 0:
            log["acciones"].append(f"Fechas inválidas detectadas: {invalid_dates}")

    # Booleana - Discount Applied
    if "Discount Applied" in df_clean.columns:
        mapping = {"True": True, "true": True, "Yes": True,
                   "False": False, "false": False, "No": False}
        df_clean["Discount Applied"] = (
            df_clean["Discount Applied"]
            .astype(str).str.strip()
            .map(mapping)
        )
        log["acciones"].append("Discount Applied convertido a booleano")

    # ── 5. Recalcular Total Spent donde sea posible ──────────────────
    if all(c in df_clean.columns for c in ["Quantity", "Price Per Unit", "Total Spent"]):
        mask_recalc = df_clean["Total Spent"].isna() & df_clean["Quantity"].notna() & df_clean["Price Per Unit"].notna()
        n_recalc = mask_recalc.sum()
        if n_recalc > 0:
            df_clean.loc[mask_recalc, "Total Spent"] = (
                df_clean.loc[mask_recalc, "Quantity"] * df_clean.loc[mask_recalc, "Price Per Unit"]
            )
            log["acciones"].append(f"Recalculados {n_recalc:,} valores de Total Spent a partir de Quantity × Price Per Unit")

        # Recuperar Price Per Unit
        mask_price = df_clean["Price Per Unit"].isna() & df_clean["Total Spent"].notna() & df_clean["Quantity"].notna() & (df_clean["Quantity"] != 0)
        n_price = mask_price.sum()
        if n_price > 0:
            df_clean.loc[mask_price, "Price Per Unit"] = (
                df_clean.loc[mask_price, "Total Spent"] / df_clean.loc[mask_price, "Quantity"]
            )
            log["acciones"].append(f"Recuperados {n_price:,} valores de Price Per Unit (Total Spent / Quantity)")

        # Recuperar Quantity
        mask_qty = df_clean["Quantity"].isna() & df_clean["Total Spent"].notna() & df_clean["Price Per Unit"].notna() & (df_clean["Price Per Unit"] != 0)
        n_qty = mask_qty.sum()
        if n_qty > 0:
            df_clean.loc[mask_qty, "Quantity"] = (
                df_clean.loc[mask_qty, "Total Spent"] / df_clean.loc[mask_qty, "Price Per Unit"]
            ).round(0).astype(int)
            log["acciones"].append(f"Recuperados {n_qty:,} valores de Quantity (Total Spent / Price Per Unit)")

    # ── 6. Tratar cantidades negativas ───────────────────────────────
    if "Quantity" in df_clean.columns:
        neg_mask = df_clean["Quantity"] < 0
        n_neg = neg_mask.sum()
        if n_neg > 0:
            log["outliers_dataframes"]["cantidades_negativas"] = df_clean[neg_mask][
                ["Transaction ID", "Quantity", "Total Spent"]
            ].copy().rename(columns={"Quantity": "Quantity_Original"})
            df_clean.loc[neg_mask, "Quantity"] = df_clean.loc[neg_mask, "Quantity"].abs()
            log["outliers_detectados"]["Quantity_Negativa"] = {
                "cantidad": int(n_neg),
                "accion": "Convertidas a valor absoluto",
            }
            log["acciones"].append(f"Convertidas {n_neg} cantidades negativas a valor absoluto")

    # ── 7. Detección y capeo de outliers ─────────────────────────────
    if treat_outliers:
        for col in ["Quantity", "Price Per Unit", "Total Spent"]:
            if col in df_clean.columns:
                mask_out, lower, upper = detect_outliers_iqr(df_clean[col], multiplier=3)
                n_out = mask_out.sum()
                if n_out > 0:
                    log["outliers_dataframes"][f"{col}_outliers"] = df_clean[mask_out][
                        [c for c in ["Transaction ID", col] if c in df_clean.columns]
                    ].copy()
                    # Capear
                    df_clean.loc[df_clean[col] > upper, col] = upper
                    df_clean.loc[df_clean[col] < lower, col] = lower
                    log["outliers_detectados"][col] = {
                        "cantidad": int(n_out),
                        "limite_inferior": round(lower, 2) if lower is not None else None,
                        "limite_superior": round(upper, 2) if upper is not None else None,
                        "accion": f"Capeados a [{lower:.2f}, {upper:.2f}]",
                    }
                    log["acciones"].append(f"Outliers en {col}: {n_out:,} valores capeados (IQR×3)")

    # ── 8. Imputación de nulos numéricos ─────────────────────────────
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        n_null = df_clean[col].isna().sum()
        if n_null > 0:
            if impute_method == "Media":
                val = df_clean[col].mean()
                df_clean[col].fillna(val, inplace=True)
            elif impute_method == "Mediana":
                val = df_clean[col].median()
                df_clean[col].fillna(val, inplace=True)
            else:
                val = 0
                df_clean[col].fillna(0, inplace=True)
            log["imputaciones"][col] = {
                "metodo": impute_method,
                "valor_imputado": round(val, 2) if isinstance(val, float) else val,
                "valores_afectados": int(n_null),
                "justificacion": f"Se usó {impute_method.lower()} para preservar la distribución central."
                if impute_method != "Cero"
                else "Se rellenó con 0 por decisión del usuario.",
            }

    # ── 9. Imputación de nulos categóricos ───────────────────────────
    cat_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        n_null = df_clean[col].isna().sum()
        if n_null > 0:
            mode_val = df_clean[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Desconocido"
            df_clean[col].fillna(fill_val, inplace=True)
            log["imputaciones"][col] = {
                "metodo": "Moda",
                "valor_imputado": fill_val,
                "valores_afectados": int(n_null),
                "justificacion": f"Se imputó con la moda ('{fill_val}') para mantener la distribución.",
            }

    # Imputar Discount Applied nulos con False
    if "Discount Applied" in df_clean.columns:
        n_disc_null = df_clean["Discount Applied"].isna().sum()
        if n_disc_null > 0:
            df_clean["Discount Applied"].fillna(False, inplace=True)
            log["imputaciones"]["Discount Applied"] = {
                "metodo": "Valor por defecto (False)",
                "valor_imputado": False,
                "valores_afectados": int(n_disc_null),
                "justificacion": "Se asume que no hubo descuento cuando el dato está ausente.",
            }

    log["registros_finales"] = len(df_clean)
    return df_clean, log


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_derived_features(df):
    """
    Crea variables derivadas para análisis avanzado.

    Features creadas:
    - Ticket_Promedio: Total Spent / Quantity
    - Mes, Dia_Semana, Trimestre, Anio: extraídos de Transaction Date
    - Periodo_Dia: Mañana / Tarde / Noche (basado en mes como proxy)
    - Rango_Gasto: Bajo / Medio / Alto / Premium (cuartiles de Total Spent)
    - Revenue_con_Descuento / Revenue_sin_Descuento: segmentación de ingreso
    """
    df_feat = df.copy()

    # Ticket Promedio
    if all(c in df_feat.columns for c in ["Total Spent", "Quantity"]):
        df_feat["Ticket_Promedio"] = np.where(
            df_feat["Quantity"] > 0,
            (df_feat["Total Spent"] / df_feat["Quantity"]).round(2),
            0,
        )

    # Temporales
    if "Transaction Date" in df_feat.columns:
        df_feat["Transaction Date"] = pd.to_datetime(df_feat["Transaction Date"], errors="coerce")
        df_feat["Mes"] = df_feat["Transaction Date"].dt.month
        df_feat["Nombre_Mes"] = df_feat["Transaction Date"].dt.month_name()
        df_feat["Dia_Semana"] = df_feat["Transaction Date"].dt.day_name()
        df_feat["Trimestre"] = df_feat["Transaction Date"].dt.quarter
        df_feat["Anio"] = df_feat["Transaction Date"].dt.year
        df_feat["Dia_Mes"] = df_feat["Transaction Date"].dt.day

        # Fin de semana
        df_feat["Es_FinDeSemana"] = df_feat["Transaction Date"].dt.dayofweek >= 5

    # Rango de Gasto (cuartiles)
    if "Total Spent" in df_feat.columns:
        df_feat["Rango_Gasto"] = pd.qcut(
            df_feat["Total Spent"],
            q=4,
            labels=["Bajo", "Medio", "Alto", "Premium"],
            duplicates="drop",
        )

    return df_feat


# =============================================================================
# REPORTE DE LIMPIEZA
# =============================================================================

def generate_cleaning_report(health_before, health_after, cleaning_log):
    """Genera un reporte estructurado de limpieza."""
    return {
        "metricas_antes": {
            "health_score": health_before["health_score"],
            "completitud": health_before["completitud_pct"],
            "unicidad": health_before["unicidad_pct"],
            "validez": health_before["validez_pct"],
            "registros": health_before["total_registros"],
            "celdas_nulas": health_before["celdas_nulas"],
            "duplicados": health_before["registros_duplicados"],
        },
        "metricas_despues": {
            "health_score": health_after["health_score"],
            "completitud": health_after["completitud_pct"],
            "unicidad": health_after["unicidad_pct"],
            "validez": health_after["validez_pct"],
            "registros": health_after["total_registros"],
            "celdas_nulas": health_after["celdas_nulas"],
            "duplicados": health_after["registros_duplicados"],
        },
        "mejora_health_score": round(
            health_after["health_score"] - health_before["health_score"], 2
        ),
        "acciones_realizadas": cleaning_log["acciones"],
        "outliers_detectados": cleaning_log["outliers_detectados"],
        "imputaciones": cleaning_log["imputaciones"],
        "nulidad_antes": health_before["nulidad_por_columna"],
        "nulidad_despues": health_after["nulidad_por_columna"],
    }
