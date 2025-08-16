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




class DataProcessingConfig:
    def __init__(self, training_pipeline_config: TrainPipelineConfig):
        self.data_processing_dir = os.path.join(training_pipeline_config.artifact_dir, trainingpipeline.DATA_PROCESSING_DIR_NAME)
        self.data_processor_obj_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PREPROCESSING_PIPELINE_OBJ_DIR)
        self.data_processor_obj_path = os.path.join(self.data_processor_obj_dir, trainingpipeline.DATA_PREPROCESSING_PIPELINE_OBJ_NAME)
        self.data_processing_dataset_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_DATALOADER_DIR_NAME)
        self.train_dataset_path = os.path.join(self.data_processing_dataset_dir, trainingpipeline.DATA_PROCESSING_TRAIN_DATASET_NAME)
        self.test_dataset_path = os.path.join(self.data_processing_dataset_dir, trainingpipeline.DATA_PROCESSING_TEST_DATASET_NAME)
        self.data_tramsform_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_DATA_TRANSFORM_DIR_NAME)
        self.transformed_X_train_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_X_TRAIN_TRANSFORM_NAME)
        self.transformed_y_train_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_Y_TRAIN_TRANSFORM_NAME)
        self.transformed_ts_train_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_TS_TRANSFORM_NAME)
        self.transformed_X_test_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_X_TEST_TRANSFORM_NAME)
        self.transformed_y_test_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_Y_TEST_TRANSFORM_NAME)
        self.transformed_ts_test_path = os.path.join(self.data_tramsform_dir, trainingpipeline.DATA_PROCESSING_TS_TEST_TRANSFORM_NAME)
        self.window_size = trainingpipeline.WINDOW_SIZE
        self.window_step = trainingpipeline.WINDOW_STEP
        self.test_split_ratio = trainingpipeline.TEST_SPLIT_RATIO
        self.batch_size = trainingpipeline.BATCH_SIZE
        self.shuffle_train = trainingpipeline.SHUFFLE_TRAIN
        self.num_workers = trainingpipeline.NUM_WORKERS
        self.pin_memory = trainingpipeline.PIN_MEMORY