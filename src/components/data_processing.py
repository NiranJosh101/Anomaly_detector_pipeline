import sys, os
import numpy as np
import pandas as pd


from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.config_entities.config_entity import DataProcessingConfig
from src.config_entities.artifact_entity import DataIngestionArtifact

from src.utils.common import detect_timestamp_column



class DataProcessing:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_processing_config: DataProcessingConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_processing_config = data_processing_config
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    
    def read_validated_data(self) -> pd.DataFrame:
        """
        Reads the validated data from the ingestion artifact.
        """
        try:
            validated_data_path = self.data_ingestion_artifact.raw_data_file_path
            if not validated_data_path:
                raise ValueError("No validated data path provided.")

            df = pd.read_csv(validated_data_path)
            logger.logging.info(f"Validated data read from {validated_data_path}")
            return df

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    def handle_missing_and_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect timestamp column and handle missing/invalid values in a time-series-friendly way.
        Returns cleaned DataFrame.
        """
        try:
            
            time_col = detect_timestamp_column(df)
            if time_col is None:
                raise ValueError("Could not detect timestamp column automatically. "
                                "Please provide a timestamp column or check the data.")

            
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).copy()

            
            df = df.sort_values(by=time_col).reset_index(drop=True)

            # Fill missing values
            if df.isna().sum().sum() > 0:
                df = df.ffill().bfill()  # forward then backward fill
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if df[numeric_cols].isna().sum().sum() > 0:
                    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit_direction="both")

                # Final drop of any still-missing critical values
                df = df.dropna(subset=[time_col]).reset_index(drop=True)

            logger.logging.info(f"Missing & invalid handling complete. Detected time column: {time_col}")
            return df

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        
    
    

    def resample_to_uniform_frequency(df: pd.DataFrame, freq: str = None, fill_method: str = "ffill") -> pd.DataFrame:
        """
        Resamples the DataFrame to a uniform frequency if needed, with optional missing value handling.

        """
        
        timestamp_col = detect_timestamp_column(df)
        if timestamp_col is None:
            raise ValueError("No valid timestamp column found for resampling.")

       
        if df.index.name != timestamp_col:
            df = df.set_index(timestamp_col)

        # Auto-detect frequency if not provided
        if freq is None:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                freq = inferred_freq
            else:
    
                deltas = df.index.to_series().diff().dropna()
                most_common_delta = deltas.mode()[0]
                freq = pd.tseries.frequencies.to_offset(most_common_delta).freqstr
                print(f"Auto-selected frequency: {freq}")

        # Resample to uniform frequency
        df_resampled = df.resample(freq).mean()

        # Fill missing values if requested
        if fill_method in ["ffill", "bfill"]:
            df_resampled = df_resampled.fillna(method=fill_method)

        return df_resampled




