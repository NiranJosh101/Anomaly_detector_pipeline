from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import sys
import shutil


from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.config_entities.config_entity import TrainPipelineConfig, DataIngestionConfig
from src.pipeline.training_pipeline import TrainingPipeline

app = FastAPI()


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "message": "👋 Welcome to the Anomaly Detection API.",
            "upload_endpoint": "/upload",
            "usage": "Send a POST request with a CSV file to /upload"
        }
    )


@app.post("/upload")
async def upload_file(request:Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        

        train_config = TrainPipelineConfig(timestamp=datetime.now())
        data_ingestion_config = DataIngestionConfig(train_config)

        upload_path = os.path.join(data_ingestion_config.data_ingestion_dir, data_ingestion_config.uploaded_dir)
        os.makedirs(upload_path, exist_ok=True)

        destination_file_path = os.path.join(upload_path, data_ingestion_config.uploaded_data_name)
        with open(destination_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        
        train_pipeline = TrainingPipeline()
        background_tasks.add_task(train_pipeline.run_pipeline)

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully.",
                "file_path": os.path.abspath(destination_file_path)
            }
        )
    except Exception as e:
        raise AnomalyDetectionException(e, sys)