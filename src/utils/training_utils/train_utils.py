import numpy as np
import torch
from torch.utils.data import Dataset
from src.exception_setup.exception import AnomalyDetectionException
import sys



class AnomalyDetectorWindowDataset(Dataset):
    """
    PyTorch Dataset for windowed time-series data (reconstruction models).
    Includes timestamps for later mapping of reconstruction errors.
    """

    def __init__(self, X_path: str, y_path: str, ts_path: str, loader_fn):
        """
        Args:
            X_path (str): Path to windowed features (.npy)
            y_path (str): Path to windowed targets (.npy)
            ts_path (str): Path to timestamps (.npy)
            loader_fn (callable): Custom function to load numpy arrays
        """
        try:
            self.X = loader_fn(X_path)      
            self.y = loader_fn(y_path)      
            self.ts = loader_fn(ts_path)   
            
            if not (len(self.X) == len(self.y) == len(self.ts)):
                raise ValueError("Mismatch in number of samples between X, y, and ts arrays")

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns:
            X_tensor: torch.FloatTensor
            y_tensor: torch.FloatTensor
            ts_array: numpy array (timestamps for the window)
        """
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        ts_array = self.ts[idx]
        
        return X_tensor, y_tensor, ts_array
    



def window_data(X_arr: np.ndarray,
                y_arr: np.ndarray,
                window_size: int,
                step_size: int,
                timestamp_index: int):
    """
    Create sliding windows for time-series data (for reconstruction models).
    
    """
    X_windows, y_windows, timestamps = [], [], []

    num_samples = X_arr.shape[0]

    for start in range(0, num_samples - window_size + 1, step_size):
        end = start + window_size

        
        X_win_features = np.delete(X_arr[start:end], timestamp_index, axis=1)
        y_win_features = np.delete(y_arr[start:end], timestamp_index, axis=1)

        
        ts_value = X_arr[end - 1, timestamp_index]

        X_windows.append(X_win_features)
        y_windows.append(y_win_features)
        timestamps.append(ts_value)

    return (
        np.array(X_windows),
        np.array(y_windows),
        np.array(timestamps)
    )
