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
        self.validated_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_VALIDATED_DIR_NAME)
        self.invalid_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_INVALID_DIR_NAME)
        self.valid_x_train_name = trainingpipeline.DATA_PROCESSING_VALID_X_TRAIN_NAME
        self.valid_y_train_name = trainingpipeline.DATA_PROCESSING_VALID_Y_TRAIN_NAME
        self.valid_x_test_name = trainingpipeline.DATA_PROCESSING_VALID_X_TEST_NAME
        self.valid_y_test_name = trainingpipeline.DATA_PROCESSING_VALID_Y_TEST_NAME
        self.scaler_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_SCALER_DIR_NAME)
        self.scaler_obj_name = trainingpipeline.DATA_PROCESSING_SCALER_OBJ_NAME
        self.window_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_WINDOW_DIR_NAME)
        self.window_config_name = trainingpipeline.DATA_PROCESSING_WINDOW_NAME
        self.data_preprocessing_obj_path = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PREPROCESSING_PIPELINE_OBJ_DIR, trainingpipeline.DATA_PREPROCESSING_PIPELINE_OBJ_NAME)
        self.data_train_arr_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_TRAIN_TEST_DIR_NAME,  trainingpipeline.DATA_PROCESSING_TRAIN_ARR)
        self.data_test_arr_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_TRAIN_TEST_DIR_NAME, trainingpipeline.DATA_PROCESSING_TEST_ARR)
        self.data_train_target_arr_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_TRAIN_TEST_DIR_NAME, trainingpipeline.DATA_PROCESSING_TRAIN_TARGET_ARR)
        self.data_test_target_arr_dir = os.path.join(self.data_processing_dir, trainingpipeline.DATA_PROCESSING_TRAIN_TEST_DIR_NAME, trainingpipeline.DATA_PROCESSING_TEST_TARGET_ARR)
        # self.transformed_train_path = os.path.join(self.data_train_test_dir, trainingpipeline.DATA_PROCESSING_TRAIN_ARR)
        # self.transformed_test_path = os.path.join(self.data_train_test_dir, trainingpipeline.DATA_PROCESSING_TEST_ARR)
        self.window_size = trainingpipeline.WINDOW_SIZE
        self.window_step = trainingpipeline.WINDOW_STEP
        # self.target_column = trainingpipeline.TARGET_COLUMN 
        # self.feature_columns = trainingpipeline.FEATURE_COLUMNS
        # self.train_split_ratio = trainingpipeline.TRAIN_SPLIT_RATIO
        self.test_split_ratio = trainingpipeline.TEST_SPLIT_RATIO
        # self.scaling_strategy = trainingpipeline.SCALING_STRATEGY
        # self.scaler_feature_range = trainingpipeline.SCALER_FEATURE_RANGE
        # self.batch_size = trainingpipeline.BATCH_SIZE
        # self.shuffle_train = trainingpipeline.SHUFFLE_TRAIN
        # self.num_workers = trainingpipeline.NUM_WORKERS
        # self.pin_memory = trainingpipeline.PIN_MEMORY