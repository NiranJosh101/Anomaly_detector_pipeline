from typing import Optional, Iterable
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_object_dtype,
)
from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger
import os, sys
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV





def detect_timestamp_column(
    df: pd.DataFrame,
    threshold: float = 0.8,
    candidate_names: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """
    Detect which column most likely contains timestamps.

    Priority:
      1) Explicit candidate_names (if provided)
      2) datetime64 columns
      3) object/string columns that parse to datetimes
      4) numeric columns that look like unix timestamps (guarded)

    Returns column name or None if confidence < threshold.
    """
    if df is None or df.shape[1] == 0:
        return None

    now = pd.Timestamp.now()
    earliest = pd.Timestamp("1900-01-01")               
    latest = now + pd.DateOffset(years=50)

    total = len(df)
    best_col = None
    best_score = 0.0

    def frac_in_range(ts: pd.Series) -> float:
        s = ts.dropna()
        if s.empty:
            return 0.0
        mask = (s >= earliest) & (s <= latest)
        return float(mask.sum()) / float(total)

    def parse_object(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    def is_mono(ts: pd.Series) -> bool:
        s = pd.Series(ts).dropna()
        # allow equality (non-decreasing) to avoid jitter issues
        return bool(s.is_monotonic_increasing or s.is_monotonic)

    # --- 0) Explicit candidates (if provided) ---
    if candidate_names:
        for name in candidate_names:
            if name in df.columns:
                col = df[name]
                if is_datetime64_any_dtype(col):
                    return name
                if col.dtype.kind in ("O", "U", "S"):
                    parsed = parse_object(col)
                    if (parsed.notna().mean() >= threshold):
                        return name
                # numeric fallback with guards
                if np.issubdtype(col.dtype, np.number):
                    series = col.dropna()
                    if not series.empty:
                        # Guard: skip obviously non-timestamp scales
                        if series.median() >= 10_000_000:  # ~1970-04 in seconds
                            s_sec = pd.to_datetime(series, unit="s", errors="coerce")
                            if (s_sec.notna().mean() >= threshold):
                                return name
                        if series.median() >= 1_000_000_000_000:  # milliseconds
                            s_ms = pd.to_datetime(series, unit="ms", errors="coerce")
                            if (s_ms.notna().mean() >= threshold):
                                return name

    # --- 1) datetime64 columns ---
    dt_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
    if dt_cols:
        # If multiple, prefer monotonic & in-range
        best = None
        best_dt_score = -1.0
        for c in dt_cols:
            col = df[c]
            score = frac_in_range(col)
            if is_mono(col):
                score += 0.2  # small bonus for monotonicity
            if score > best_dt_score:
                best_dt_score = score
                best = c
        return best

    # --- 2) object/string columns ---
    for c in df.select_dtypes(include=["object", "string"]).columns:
        try:
            parsed = parse_object(df[c])
            score = parsed.notna().mean()
            # prefer in-range & monotonic
            score += 0.2 * frac_in_range(parsed)
            if is_mono(parsed):
                score += 0.1
        except Exception:
            score = 0.0

        if score > best_score:
            best_score = score
            best_col = c

    if best_col and best_score >= threshold:
        return best_col

    # --- 3) numeric unix timestamps (very conservative) ---
    for c in df.select_dtypes(include=["number"]).columns:
        series = df[c].dropna()
        if series.empty:
            continue

        # Guard against small integers (counts, IDs, etc.)
        med = float(series.median())

        score = 0.0
        # Seconds since epoch
        if med >= 10_000_000:  # ~1970-04 in seconds
            s_sec = pd.to_datetime(series, unit="s", errors="coerce")
            score = max(score, s_sec.notna().mean() + 0.2 * frac_in_range(s_sec) + (0.1 if is_mono(s_sec) else 0.0))

        # Milliseconds since epoch
        if med >= 1_000_000_000_000:  # ~2001 in ms
            s_ms = pd.to_datetime(series, unit="ms", errors="coerce")
            score = max(score, s_ms.notna().mean() + 0.2 * frac_in_range(s_ms) + (0.1 if is_mono(s_ms) else 0.0))

        if score > best_score:
            best_score = score
            best_col = c

    if best_col and best_score >= threshold:
        return best_col

    return None




def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
    


def save_object(file_path: str, obj: object) -> None:
    try:
        logger.logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
    


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
    


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise AnomalyDetectionException(e, sys) from e
