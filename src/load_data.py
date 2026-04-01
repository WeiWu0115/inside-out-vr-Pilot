"""
Data loading and validation for the multi-agent pipeline.
"""

import sys
import pandas as pd
from config import REQUIRED_COLUMNS, OPTIONAL_COLUMNS


def load_csv(path: str) -> pd.DataFrame:
    """Load the window-level CSV and validate columns."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Input file not found: {path}")
    except Exception as e:
        sys.exit(f"[ERROR] Could not read CSV: {e}")

    if df.empty:
        sys.exit("[ERROR] Input CSV is empty.")

    # Check required columns
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        sys.exit(
            f"[ERROR] Missing required columns: {missing_required}\n"
            f"  Found columns: {list(df.columns)}"
        )

    # Warn about missing optional columns
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]
    if missing_optional:
        print(f"[WARN] Missing optional columns (agents will output 'unknown'): {missing_optional}")

    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def safe_get(row: pd.Series, col: str, default=None):
    """Safely retrieve a value from a row, returning default if missing or NaN."""
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val
