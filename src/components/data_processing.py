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
from src.config_entities.artifact_entity import DataIngestionArtifact, DataProcessingArtifact

from src.utils.common import detect_timestamp_column,save_object,save_numpy_array_data,load_object,load_numpy_array_data
from src.utils.training_utils.train_utils import window_data



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
        
    


    def resampling_df_freqencies(self, df: pd.DataFrame) -> pd.DataFrame:
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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform a chronological train/test split for time-series data.
        For reconstruction models: y == X.
      
        """
        try:
            time_col = detect_timestamp_column(df)
            if time_col is None:
                raise ValueError("Timestamp column could not be detected.")

            df = df.sort_values(by=time_col).reset_index(drop=True)

            split_index = int(len(df) * (1 - self.data_processing_config.test_split_ratio))

            X_train = df.iloc[:split_index].reset_index(drop=True)
            X_test = df.iloc[split_index:].reset_index(drop=True)

            # y_train = X_train.copy()
            # y_test = X_test.copy()

            logger.logging.info(
                f"Time-series split complete. "
                f"Train size: {len(X_train)}, Test size: {len(X_test)}"
            )

            return X_train, X_test

        except Exception as e:
            raise AnomalyDetectionException(e, sys)

            
    
    def fit_and_save_preprocessing_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """
        Fit the preprocessing pipeline (scaling + encoding + timestamp passthrough)
        on training data only, save it for later inference.
        
        Args:
            X_train: Training DataFrame (before scaling/encoding)
            
        Returns:
            The fitted preprocessing pipeline.
        """
        try:
            # 1. Identify column types
            numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
            datetime_cols = X_train.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

            if not datetime_cols:
                raise ValueError("No datetime column found in dataset.")

            timestamp_col = datetime_cols[0]  # assume first datetime col is timestamp

            # 2. Pipelines for numeric and categorical
            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            # 3. ColumnTransformer: pass timestamps through without scaling/encoding
            preprocessing = ColumnTransformer([
                ("num", num_pipeline, numeric_cols),
                ("cat", cat_pipeline, categorical_cols),
                ("time", "passthrough", [timestamp_col])  # keep timestamps
            ], remainder="drop")

            # 4. Fit on training data
            preprocessing.fit(X_train)

            # 5. Save fitted preprocessing object
            save_object(
                file_path=self.data_processing_config.preprocessing_object_path,
                obj=preprocessing
            )

            logger.logging.info("Preprocessing pipeline fitted and saved.")

            return preprocessing

        except Exception as e:
            raise AnomalyDetectionException(e, sys)

            
            
    def transform_train_test(self, X_train, X_test):
        try:
            pipeline_path = self.data_processing_config.data_preprocessing_obj_path
            preprocessor = load_object(pipeline_path)

            # Transform datasets (outputs X only)
            X_train_arr = preprocessor.transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            # For reconstruction models: y is identical to X
            y_train_arr = X_train_arr.copy()
            y_test_arr = X_test_arr.copy()

            # Save transformed datasets
            save_numpy_array_data(
                file_path=self.data_processing_config.data_train_arr_dir,
                array=X_train_arr
            )
            save_numpy_array_data(
                file_path=self.data_processing_config.data_test_arr_dir,
                array=X_test_arr
            )
            save_numpy_array_data(
                file_path=self.data_processing_config.data_train_target_arr_dir,
                array=y_train_arr
            )
            save_numpy_array_data(
                file_path=self.data_processing_config.data_test_target_arr_dir,
                array=y_test_arr
            )

            logger.logging.info("Train and test datasets transformed (X & y) and saved.")
            return X_train_arr, y_train_arr, X_test_arr, y_test_arr

        except Exception as e:
            raise AnomalyDetectionException(e, sys)


    def apply_windowing_to_train_test(self,
                                  X_train_arr: np.ndarray,
                                  y_train_arr: np.ndarray,
                                  X_test_arr: np.ndarray,
                                  y_test_arr: np.ndarray):
        """
        Apply windowing to both train and test sets for reconstruction models,
        with safe timestamp column detection and fallback.
        """
        try:
            # Step 1: Try to auto-detect timestamp index
            timestamp_index = None
            for col_idx in range(X_train_arr.shape[1]):
                if not np.issubdtype(X_train_arr[:, col_idx].dtype, np.number):
                    timestamp_index = col_idx
                    logger.logging.info(f"Auto-detected timestamp column at index {timestamp_index}")
                    break
            
            # Step 2: Fallback to config if auto-detection fails
            if timestamp_index is None:
                if hasattr(self.data_processing_config, "timestamp_index"):
                    timestamp_index = self.data_processing_config.timestamp_index
                    logger.logging.warning(
                        f"Auto-detection failed. Using fallback timestamp index from config: {timestamp_index}"
                    )
                else:
                    raise ValueError(
                        "No non-numeric column found for timestamp detection and no fallback index provided."
                    )

            # Step 3: Apply windowing
            X_train_win, y_train_win, ts_train = window_data(
                X_train_arr, y_train_arr,
                window_size=self.data_processing_config.window_size,
                step_size=self.data_processing_config.window_step,
                timestamp_index=timestamp_index
            )

            X_test_win, y_test_win, ts_test = window_data(
                X_test_arr, y_test_arr,
                window_size=self.data_processing_config.window_size,
                step_size=self.data_processing_config.window_step,
                timestamp_index=timestamp_index
            )

            # Step 4: Save results
            save_numpy_array_data(self.data_processing_config.windowed_X_train_path, X_train_win)
            save_numpy_array_data(self.data_processing_config.windowed_y_train_path, y_train_win)
            save_numpy_array_data(self.data_processing_config.windowed_ts_train_path, ts_train)

            save_numpy_array_data(self.data_processing_config.windowed_X_test_path, X_test_win)
            save_numpy_array_data(self.data_processing_config.windowed_y_test_path, y_test_win)
            save_numpy_array_data(self.data_processing_config.windowed_ts_test_path, ts_test)

            logger.logging.info("Windowing applied to train & test sets and saved successfully.")

            return X_train_win, y_train_win, ts_train, X_test_win, y_test_win, ts_test

        except Exception as e:
            raise AnomalyDetectionException(e, sys)



        
    def initiate_data_processing(self) -> DataProcessingArtifact:
        """
        Main method to handle the entire data processing pipeline.
        """
        try:

            df = self.read_validated_data()
            df = self.handle_missing_and_invalid(df)
            df = self.resampling_df_freqencies(df)
            train_df, test_df = self.train_test_split_time_series(
                df, 
                self.data_processing_config.test_split_ratio
            )
            preprocessor = self.fit_and_save_preprocessing_pipeline(train_df)
            X_train, y_train, X_test, y_test = self.transform_train_test(train_df, test_df)

    

            return DataProcessingArtifact(
                X_train_path=self.data_processing_config.data_train_arr_dir,
                y_train_path=self.data_processing_config.data_train_target_arr_dir,
                X_test_path=self.data_processing_config.data_test_arr_dir,
                y_test_path=self.data_processing_config.data_test_target_arr_dir,
                preprocessor=self.data_processing_config.data_preprocessing_obj_path
            )

        except Exception as e:
            raise AnomalyDetectionException(e, sys)
