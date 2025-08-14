import os
import sys

from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.components.data_ingestion import DataIngestion
from src.components.data_processing import DataProcessing

from src.config_entities.artifact_entity import DataIngestionArtifact, DataProcessingArtifact


from src.config_entities.config_entity import (
    DataIngestionConfig,
    DataProcessingConfig,
    TrainPipelineConfig,
)



class TrainingPipeline:
    def __init__(self):
        try:
            self.trainingpipeline_config = TrainPipelineConfig()
        except Exception as e:
            raise AnomalyDetectionException(e, sys)

    def start_data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.trainingpipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config)
            logger.logging.info("<<======Initiate the data ingestion=======>>")
            dataingestionartifact= data_ingestion.initiate_data_ingestion()
            logger.logging.info("<<======Data Ingestion Complete=======>>")
            return dataingestionartifact
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    def start_data_processing(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_processing_config = DataProcessingConfig(training_pipeline_config=self.trainingpipeline_config)
            data_processing = DataProcessing(data_ingestion_artifact, data_processing_config)
            logger.logging.info("<<======Initiate the data processing=======>>")
            data_processing_artifact = data_processing.initiate_data_processing()
            logger.logging.info("<<======Data Processing Complete=======>>")
            return data_processing_artifact
        except Exception as e:
            raise AnomalyDetectionException(e, sys)
        

    

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_processing_artifact = self.start_data_processing(data_ingestion_artifact)
            # return data_ingestion_artifact
        except Exception as e:
            raise AnomalyDetectionException(e, sys)