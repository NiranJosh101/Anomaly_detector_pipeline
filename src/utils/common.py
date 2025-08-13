from typing import Optional
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



def detect_timestamp_column(df: pd.DataFrame, threshold: float = 0.8) -> Optional[str]:
    """
    Detect which column most likely contains timestamps.

    """
    if df is None or df.shape[1] == 0:
        return None

    now = pd.Timestamp.now()
    earliest = pd.Timestamp("1970-01-01")
    latest = now + pd.DateOffset(years=5)

    best_col = None
    best_frac = 0.0

    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            return col 

    total = len(df)

    def valid_frac(ts_series: pd.Series) -> float:
        if ts_series.isna().all():
            return 0.0
        in_range = ts_series.dropna().apply(lambda x: (x >= earliest) and (x <= latest))
        return in_range.sum() / total

    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if series.empty:
            continue

        try:
            s_sec = pd.to_datetime(series, unit="s", errors="coerce")
            frac_sec = valid_frac(s_sec)
        except Exception:
            frac_sec = 0.0


        try:
            s_ms = pd.to_datetime(series, unit="ms", errors="coerce")
            frac_ms = valid_frac(s_ms)
        except Exception:
            frac_ms = 0.0

        col_best_frac = max(frac_sec, frac_ms)
        if col_best_frac > best_frac:
            best_frac = col_best_frac
            best_col = col

    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            frac = parsed.notna().sum() / total
            
            frac_in_range = valid_frac(parsed)

            metric = frac_in_range
        except Exception:
            metric = 0.0

        if metric > best_frac:
            best_frac = metric
            best_col = col

    if best_frac >= threshold:
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
