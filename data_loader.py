import os
import pandas as pd
import numpy as np


def detect_column_info(df: pd.DataFrame, col: str) -> dict:
    """
    Smart column analysis — detects dtype, missing values, outliers & cardinality.
    This powers the 'Ultimate' cleaning engine.
    """
    series = df[col]
    total = len(series)
    missing = int(series.isnull().sum())
    not_null = series.dropna()
    unique_count = int(not_null.nunique())

    # ── Dtype Detection ────────────────────────────────────────────────────────
    if pd.api.types.is_integer_dtype(series):
        col_type = "numeric"
        original_dtype = "int"
    elif pd.api.types.is_float_dtype(series):
        col_type = "numeric"
        original_dtype = "float"
    else:
        # Try datetime detection
        if not_null.dtype == object and len(not_null) > 0:
            try:
                pd.to_datetime(not_null.head(30), infer_datetime_format=True, errors="raise")
                col_type = "datetime"
                original_dtype = "datetime"
            except Exception:
                col_type = "categorical"
                original_dtype = "categorical"
        else:
            col_type = "categorical"
            original_dtype = "categorical"

    # ── Outlier Detection (IQR method, numeric only) ───────────────────────────
    has_outliers = False
    if col_type == "numeric" and len(not_null) >= 4:
        Q1 = float(not_null.quantile(0.25))
        Q3 = float(not_null.quantile(0.75))
        IQR = Q3 - Q1
        if IQR > 0:
            n_outliers = int(((not_null < Q1 - 1.5 * IQR) | (not_null > Q3 + 1.5 * IQR)).sum())
            has_outliers = n_outliers > 0

    return {
        "type": col_type,
        "original_dtype": original_dtype,
        "missing_values": missing,
        "unique_count": unique_count,
        "has_outliers": has_outliers,
    }


def df_to_state(df: pd.DataFrame, base_accuracy: float = 0.5) -> dict:
    """Pandas DataFrame → OpenEnv state format (with rich column metadata)."""
    state = {
        "total_rows": len(df),
        "current_accuracy": base_accuracy,
        "columns": {}
    }
    for col in df.columns:
        state["columns"][col] = detect_column_info(df, col)
    return state


def load_task_state(file_path: str, fallback_state: dict) -> dict:
    """Load file if it exists, otherwise return fallback dummy state."""
    if os.path.exists(file_path):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                return fallback_state
            return df_to_state(df, fallback_state.get("current_accuracy", 0.5))
        except Exception as e:
            print(f"File load error ({file_path}): {e}. Using fallback.")
            return fallback_state
    return fallback_state