from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class WindowingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def fit(self, X, y=None):
        return self  # Nothing to learn

    def transform(self, X):
        X_win = []
        for i in range(0, len(X) - self.window_size + 1, self.step_size):
            X_win.append(X[i:i+self.window_size])
        return np.array(X_win)

