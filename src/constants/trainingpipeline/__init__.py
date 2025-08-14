import os
import sys
import numpy as np
import pandas as pd


PIPELINE_NAME = "ANnomalyDetectionPipeline"
ARTIFACT_DIR_NAME = "Artifact"
DATA_DIR_NAME = "data"


DATA_UPLOADED_NAME = "UploadedData.csv"
DATA_PROCESSED_NAME = "ProcessedData.csv"
DATA_PROCESS_COMPLETED_NAME = "ValidatedData.csv"
DATA_PROCESS_FAILED_NAME = "InvalidData.csv"
DATA_UPLOADED_DIR_NAME = "Uploaded"
DATA_PROCESSED_DIR_NAME = "Processed"
DATA_PROCESS_COMPLETED_DIR_NAME = "valid_dataset"
DATA_PROCESS_FAILED_DIR_NAME = "invalid_dataset"


DATA_PROCESSING_DIR_NAME = "DataProcessing"
DATA_PROCESSING_VALIDATED_DIR_NAME = "Validated"
DATA_PROCESSING_INVALID_DIR_NAME = "Invalid"
DATA_PROCESSING_VALID_X_TRAIN_NAME = "X_train.csv"
DATA_PROCESSING_VALID_Y_TRAIN_NAME = "y_train.csv"
DATA_PROCESSING_VALID_X_TEST_NAME = "X_test.csv"
DATA_PROCESSING_VALID_Y_TEST_NAME = "y_test.csv"
DATA_PROCESSING_SCALER_DIR_NAME = "scaler obj"
DATA_PROCESSING_SCALER_OBJ_NAME = "scaler.pkl"
DATA_PROCESSING_WINDOW_DIR_NAME = "window"
DATA_PROCESSING_WINDOW_NAME = "window.ymal"
DATA_PREPROCESSING_PIPELINE_OBJ_DIR = "preprocessing_obj"
DATA_PREPROCESSING_PIPELINE_OBJ_NAME = "preprocessing_pipeline.pkl"
DATA_PROCESSING_TRAIN_TEST_DIR_NAME = "transformed_data"
DATA_PROCESSING_TRAIN_ARR = "train_data.npy"
DATA_PROCESSING_TEST_ARR = "test_data.npy"
DATA_PROCESSING_TRAIN_TARGET_ARR = "train_target.npy"
DATA_PROCESSING_TEST_TARGET_ARR = "test_target.npy"
WINDOW_SIZE = 30            
WINDOW_STEP = 1             
TARGET_COLUMN = "value"     
FEATURE_COLUMNS = None
TRAIN_SPLIT_RATIO = 0.8
TEST_SPLIT_RATIO = 0.2
SCALING_STRATEGY = "minmax"   # Options: 'minmax', 'standard', etc.
SCALER_FEATURE_RANGE = (0, 1) # Only for minmax scaling
BATCH_SIZE = 32
SHUFFLE_TRAIN = True
NUM_WORKERS = 2
PIN_MEMORY = True



