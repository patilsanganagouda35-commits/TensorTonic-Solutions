import numpy as np

def mean_squared_error(y_pred, y_true):
    y_pred = np.array(y_pred, dtype=np.float64)
    y_true = np.array(y_true, dtype=np.float64)
    
    return float(np.mean((y_pred - y_true) ** 2))