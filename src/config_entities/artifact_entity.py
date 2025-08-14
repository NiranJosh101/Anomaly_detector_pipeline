from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    raw_data_file_path: str



@dataclass
class DataProcessingArtifact:
    X_train_path: str
    y_train_path: str
    X_test_path: str
    y_test_path: str
    preprocessor: str

    