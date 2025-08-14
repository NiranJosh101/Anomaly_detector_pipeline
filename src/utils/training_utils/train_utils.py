from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class WindowingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size, timestamp_index=None):
        self.window_size = window_size
        self.step_size = step_size
        self.timestamp_index = timestamp_index  # Index/column for timestamp in X

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        X: np.ndarray or DataFrame -> preprocessed features + timestamp column
        Returns: X_windows, y_windows, timestamps
        """
        X_windows, y_windows, timestamps = [], [], []

        # If X is a DataFrame, convert to numpy but keep timestamps separately
        if hasattr(X, "iloc") and self.timestamp_index is not None:
            timestamps_array = X.iloc[:, self.timestamp_index].values
            X_numeric = X.drop(X.columns[self.timestamp_index], axis=1).values
        else:
            timestamps_array = X[:, self.timestamp_index] if self.timestamp_index is not None else None
            X_numeric = np.delete(X, self.timestamp_index, axis=1) if self.timestamp_index is not None else X

        for i in range(0, len(X_numeric) - self.window_size + 1, self.step_size):
            X_win = X_numeric[i:i + self.window_size, :]
            X_windows.append(X_win)
            y_windows.append(X_win)

            if timestamps_array is not None:
                timestamps.append(timestamps_array[i:i + self.window_size])

        return (
            np.array(X_windows, dtype=np.float32),
            np.array(y_windows, dtype=np.float32),
            np.array(timestamps) if timestamps_array is not None else None
        )
