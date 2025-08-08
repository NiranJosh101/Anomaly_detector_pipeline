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