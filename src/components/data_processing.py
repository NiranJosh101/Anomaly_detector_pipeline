import sys, os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer




from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.config_entities.config_entity import DataProcessingConfig
from src.config_entities.artifact_entity import DataIngestionArtifact

from src.utils.common import detect_timestamp_column,save_object,save_numpy_array_data,load_object,load_numpy_array_data
from src.utils.training_utils.train_utils import WindowingTransformer



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
        Detect timestamp column and handle missing/invalid values + duplicates in a time-series-friendly way.
        Returns cleaned DataFrame.
        """
        try:
            # Detect timestamp column
            time_col = detect_timestamp_column(df)
            if time_col is None:
                raise ValueError(
                    "Could not detect timestamp column automatically. "
                    "Please provide a timestamp column or check the data."
                )

            
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).copy()

           
            df = df.sort_values(by=time_col).reset_index(drop=True)

            # Remove duplicates 
            before_dup = len(df)
            df = df.drop_duplicates(subset=[time_col], keep="first")
            after_dup = len(df)
            if before_dup != after_dup:
                logger.logging.info(f"Removed {before_dup - after_dup} duplicate rows based on timestamp.")

            # Handle missing values
            if df.isna().sum().sum() > 0:
                
                df = df.ffill().bfill()

                numeric_cols = df.select_dtypes(include=["number"]).columns
                if df[numeric_cols].isna().sum().sum() > 0:
                    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit_direction="both")

    
                df = df.dropna(subset=[time_col]).reset_index(drop=True)

            logger.logging.info(f"Missing, invalid values, and duplicates handled. Time column: {time_col}")
            return df

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        
    


    def resampling_df_freqencies(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect timestamp column, check if data needs resampling,
        and resample to inferred frequency if irregular.
        """
        try:
            
            time_col = detect_timestamp_column(df)
            if time_col is None:
                raise ValueError("No timestamp column found.")

           
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

            # Try to infer frequency
            inferred_freq = pd.infer_freq(df[time_col])

            if inferred_freq is None:
                # If pandas can't infer, estimate it manually
                deltas = df[time_col].diff().dropna().value_counts()
                most_common_delta = deltas.index[0]
                inferred_freq = pd.tseries.frequencies.to_offset(most_common_delta).freqstr

            # Checking for irregularities
            expected_range = pd.date_range(
                start=df[time_col].min(),
                end=df[time_col].max(),
                freq=inferred_freq
            )
            is_irregular = len(expected_range) != len(df)

            if is_irregular:
                logger.logging.info(f"Data is irregular. Resampling to frequency '{inferred_freq}'")
                df = df.set_index(time_col).resample(inferred_freq).ffill().reset_index()
            else:
                logger.logging.info("Data is regular. No resampling needed.")

            return df

        except Exception as e:
            raise RuntimeError(f"Error in maybe_resample: {e}")

            

        
    def train_test_split_time_series(
            self, df: pd.DataFrame, test_size: float
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            """
            Perform a chronological train/test split for time-series data.
       
            Returns:
                (train_df, test_df)
            """
            try:
                time_col = detect_timestamp_column(df)
                if time_col is None:
                    raise ValueError("Timestamp column could not be detected.")

                df = df.sort_values(by=time_col).reset_index(drop=True)

                split_index = int(len(df) * (1 - self.data_processing_config.test_split_ratio))
                train_df = df.iloc[:split_index].reset_index(drop=True)
                test_df = df.iloc[split_index:].reset_index(drop=True)

                logger.logging.info(
                    f"Time-series split complete. Train size: {len(train_df)}, Test size: {len(test_df)}"
                )

                return train_df, test_df

            except Exception as e:
                raise AnomalyDetectionException(e, sys)
            
    
    def fit_and_save_preprocessing_pipeline(self, train_df):
        try:
            categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
            numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist()

            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            full_pipeline = Pipeline([
                ("preprocessing", ColumnTransformer([
                    ("num", num_pipeline, numeric_cols),
                    ("cat", cat_pipeline, categorical_cols)
                ], remainder="passthrough")),
                
                # Windowing AFTER preprocessing
                ("windowing", WindowingTransformer(window_size=self.data_processing_config.window_size, step_size=self.data_processing_config.window_step))
            ])

            full_pipeline.fit(train_df)

            save_object(
                file_path=os.path.join(self.data_processing_config.data_preprocessing_obj_path),
                obj=full_pipeline
            )
            logger.logging.info(f"Preprocessing pipeline saved")

            return full_pipeline

        except Exception as e:
            raise AnomalyDetectionException(e, sys)

        
        
    def transform_train_test(self, train_df, test_df):
        try:
            
            pipeline_path = self.data_processing_config.data_preprocessing_obj_path
            preprocessor = load_object(pipeline_path)

            
            X_train = preprocessor.transform(train_df)
            X_test = preprocessor.transform(test_df)

            # Save transformed datasets
            save_numpy_array_data(
                file_path=self.data_processing_config.data_train_arr_dir,
                array=X_train
            )
            save_numpy_array_data(
                file_path=self.data_processing_config.data_test_arr_dir,
                array=X_test
            )

            logger.logging.info("Train and test datasets transformed and saved.")
            return X_train, X_test

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        
    

    def create_sliding_windows(data, window_size, step_size=1, target_index=None):
        """
        data: np.ndarray (2D) -> transformed dataset
        window_size: int -> number of time steps per window
        step_size: int -> how far to move the window each step
        target_index: int or None -> column index for target variable (if supervised)
        """
        X, y = [], []
        for i in range(0, len(data) - window_size + 1, step_size):
            X.append(data[i:i + window_size, :])
            if target_index is not None:
                y.append(data[i + window_size - 1, target_index])

        X = np.array(X)
        y = np.array(y) if target_index is not None else None
        return X, y


    