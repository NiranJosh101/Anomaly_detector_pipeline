from datetime import datetime
import os
from src.constants import trainingpipeline


class TrainPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
     
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = trainingpipeline.PIPELINE_NAME
        self.artifact_name: str = trainingpipeline.ARTIFACT_DIR_NAME
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp_str)

        self.model_dir: str = os.path.join("final_model")

        self.data_dir_name: str = trainingpipeline.DATA_DIR_NAME
        self.data_dir: str = os.path.join(self.data_dir_name)

        self.timestamp: str = timestamp_str


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainPipelineConfig):

        # Base directory for data ingestion
        self.data_ingestion_dir = os.path.abspath(training_pipeline_config.data_dir)
        self.timestamp = training_pipeline_config.timestamp
        self.uploaded_dir = os.path.join(self.data_ingestion_dir, trainingpipeline.DATA_UPLOADED_DIR_NAME)
        self.processed_dir = os.path.join(self.data_ingestion_dir, trainingpipeline.DATA_PROCESSED_DIR_NAME)
        self.completed_dir = os.path.join(self.processed_dir, trainingpipeline.DATA_PROCESS_COMPLETED_DIR_NAME)
        self.failed_dir = os.path.join(self.processed_dir, trainingpipeline.DATA_PROCESS_FAILED_DIR_NAME)

        self.uploaded_data_name = trainingpipeline.DATA_UPLOADED_NAME
        self.processed_data_name = trainingpipeline.DATA_PROCESSED_NAME
        self.valid_dataset_name = trainingpipeline.DATA_PROCESS_COMPLETED_NAME
        self.invalid_dataset_name = trainingpipeline.DATA_PROCESS_FAILED_DIR_NAME
        self.invalid_csv_name = trainingpipeline.DATA_PROCESS_FAILED_NAME