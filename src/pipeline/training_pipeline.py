import os
import sys

from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.components.data_ingestion import DataIngestion


from src.config_entities.config_entity import (
    DataIngestionConfig,
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
        

    

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            # return data_ingestion_artifact
        except Exception as e:
            raise AnomalyDetectionException(e, sys)