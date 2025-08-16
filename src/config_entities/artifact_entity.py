from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_data_file_path: str



@dataclass
class DataProcessingArtifact:
    train_dataset_path: str
    test_dataset_path: str
    preprocessor: str

    