import numpy as np

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
