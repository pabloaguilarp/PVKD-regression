import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def voxel_to_point_wise(predictions, voxel_grid, batch_index):
    return predictions[batch_index, voxel_grid[batch_index][:, 0], voxel_grid[batch_index][:, 1], voxel_grid[batch_index][:, 2]]

def occupancy_hist(grid, data):
    unique_rows = np.unique(grid, axis=0)
    occupancy_values = np.empty((len(unique_rows),), dtype=np.float64)
    for i in range(len(unique_rows)):
        occupancy_values[i] = data[unique_rows[i, 0]][unique_rows[i, 1]][unique_rows[i, 2]]

def non_empty_hist(grid, data):
    data_hist = torch.empty((len(grid), ), dtype=torch.float64)
    for i in range(len(grid)):
        point_int = data[grid[i, 0]][grid[i, 1]][grid[i, 2]]
        data_hist[i] = point_int
    return data_hist

def fast_occupancy(grid, data):
    indices = tuple(grid.T.long())
    return data[indices]


################ Metrics ################

from abc import ABC, abstractmethod

# Abstract class
class Metrics(ABC):
    def __init__(self):
        self.metrics_dict = self.get_metrics_dict()

    def clear(self):
        for key, value in self.metrics_dict.items():
            self.metrics_dict[key].clear()

    def print(self):
        for key, value in self.metrics_dict.items():
            print(f"{key}: {np.mean(value)}")

    @abstractmethod
    def mean_absolute_error(self, y_true, y_pred):
        pass

    @abstractmethod
    def mean_squared_error(self, y_true, y_pred):
        pass

    @abstractmethod
    def root_mean_squared_error(self, y_true, y_pred):
        pass

    @abstractmethod
    def r2_score(self, y_true, y_pred):
        pass

    @abstractmethod
    def mean_absolute_percentage_error(self, y_true, y_pred):
        pass

    @abstractmethod
    def compute_metrics(self, y_true, y_pred, update_internal_dict):
        pass

    def get_metrics_dict(self):
        return {
            "mse": [],
            "mae": [],
            "rmse": [],
            "r2": [],
            "mape": []
        }

# NumPy implementation
class MetricsNumpy(Metrics):
    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(self.mean_squared_error(y_true, y_pred))

    def r2_score(self, y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        residual_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def compute_metrics(self, y_true, y_pred, update_internal_dict=True):
        # Compute residuals and mean of y_true once
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        mean_y_true = np.mean(y_true)

        # Calculate metrics based on precomputed values
        mae = np.mean(abs_residuals)
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum(residuals ** 2) / np.sum((y_true - mean_y_true) ** 2))
        mape = np.mean(abs_residuals / y_true) * 100

        if update_internal_dict:
            self.metrics_dict["mse"].append(mse)
            self.metrics_dict["mae"].append(mae)
            self.metrics_dict["rmse"].append(rmse)
            self.metrics_dict["r2"].append(r2)
            self.metrics_dict["mape"].append(mape)

        return mae, mse, rmse, r2, mape

# PyTorch implementation
class MetricsTorch(Metrics):
    def mean_absolute_error(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    def mean_squared_error(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def root_mean_squared_error(self, y_true, y_pred):
        return torch.sqrt(self.mean_squared_error(y_true, y_pred))

    def r2_score(self, y_true, y_pred):
        total_variance = torch.sum((y_true - torch.mean(y_true)) ** 2)
        residual_variance = torch.sum((y_true - y_pred) ** 2)
        return 1 - (residual_variance / total_variance)

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

    def compute_metrics(self, y_true, y_pred, update_internal_dict=True):
        residuals = y_true - y_pred
        abs_residuals = torch.abs(residuals)
        mean_y_true = torch.mean(y_true)

        mae = torch.mean(abs_residuals)
        mse = torch.mean(residuals ** 2)
        rmse = torch.sqrt(mse)
        r2 = 1 - (torch.sum(residuals ** 2) / torch.sum((y_true - mean_y_true) ** 2))
        mape = torch.mean(abs_residuals / y_true) * 100

        if update_internal_dict:
            self.metrics_dict["mse"].append(mse)
            self.metrics_dict["mae"].append(mae)
            self.metrics_dict["rmse"].append(rmse)
            self.metrics_dict["r2"].append(r2)
            self.metrics_dict["mape"].append(mape)

        return mae, mse, rmse, r2, mape

# Factory function to create the appropriate metrics class based on the backend
def get_metrics(backend='numpy'):
    if backend == 'numpy':
        return MetricsNumpy()
    elif backend == 'torch':
        return MetricsTorch()
    else:
        raise ValueError("Unsupported backend. Choose 'numpy' or 'torch'.")

def compute_classwise_metrics(y_true: Union[torch.Tensor, np.ndarray], y_pred: Union[torch.Tensor, np.ndarray], class_array: Union[torch.Tensor, np.ndarray], engine: str = "numpy"):
    if engine == 'numpy':
        assert isinstance(y_true, np.ndarray), "y_true must be of type np.ndarray in order to be computed with 'numpy' engine!"
        assert isinstance(y_pred, np.ndarray), "y_pred must be of type np.ndarray in order to be computed with 'numpy' engine!"
        assert isinstance(class_array, np.ndarray), "class_array must be of type np.ndarray in order to be computed with 'numpy' engine!"
        return compute_classwise_metrics_numpy(y_true, y_pred, class_array)
    elif engine == 'torch':
        assert isinstance(y_true, torch.Tensor), "y_true must be of type torch.Tensor in order to be computed with 'torch' engine!"
        assert isinstance(y_pred, torch.Tensor), "y_pred must be of type torch.Tensor in order to be computed with 'torch' engine!"
        assert isinstance(class_array, torch.Tensor), "class_array must be of type torch.Tensor in order to be computed with 'torch' engine!"
        return compute_classwise_metrics_torch(y_true, y_pred, class_array)
    else:
        print("Unsupported engine. Choose 'numpy' or 'torch'.")

def compute_classwise_metrics_torch(predictions: torch.Tensor, targets: torch.Tensor, class_indices: torch.Tensor):
    """
    Computes metrics class-wise (MSE, MAE, RMSE, R2) efficiently.

    Args:
    - predictions (torch.Tensor): Predicted values (same shape as targets).
    - targets (torch.Tensor): Ground truth values.
    - class_indices (torch.Tensor): A tensor with class indices (same shape as predictions/targets).

    Returns:
    - dict: A dictionary containing class-wise metrics (MSE, MAE, RMSE, R2).
    """
    # Ensure predictions, targets, and class_indices are of the same shape
    assert predictions.shape == targets.shape == class_indices.shape, "Shape mismatch!"

    # Get the unique classes in class_indices
    unique_classes = torch.unique(class_indices)

    # Initialize a dictionary to hold metrics for each class
    class_metrics = {}

    # Iterate through each unique class
    for cls in unique_classes:
        # Create a mask for the current class
        mask = class_indices == cls

        # Apply the mask to filter out the elements corresponding to the current class
        class_predictions = predictions[mask]
        class_targets = targets[mask]

        # Skip the class if there are no elements (e.g., if a class is absent)
        if class_predictions.numel() == 0:
            continue

        # Calculate the metrics for this class
        mse = F.mse_loss(class_predictions, class_targets)
        mae = F.l1_loss(class_predictions, class_targets)
        rmse = torch.sqrt(mse)

        # R2 (Coefficient of Determination)
        mean_target = torch.mean(class_targets)
        total_variance = torch.sum((class_targets - mean_target) ** 2)
        residual_variance = torch.sum((class_targets - class_predictions) ** 2)
        r2 = 1 - (residual_variance / total_variance) if total_variance != 0 else torch.tensor(0.0)
        epsilon = 1e-8  # To prevent division by zero
        mape = torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100

        # Store the metrics for this class in the dictionary
        class_metrics[int(cls.item())] = {
            "mse": mse.item(),
            "mae": mae.item(),
            "rmse": rmse.item(),
            "r2": r2.item(),
            "mape": mape.item(),
        }

    return class_metrics

def compute_classwise_metrics_numpy(predictions: np.ndarray, targets: np.ndarray, class_indices: np.ndarray):
    """
    Computes metrics class-wise (MSE, MAE, RMSE, R2) efficiently for NumPy arrays.

    Args:
    - predictions (np.ndarray): Predicted values (same shape as targets).
    - targets (np.ndarray): Ground truth values.
    - class_indices (np.ndarray): A numpy array with class indices (same shape as predictions/targets).

    Returns:
    - dict: A dictionary containing class-wise metrics (MSE, MAE, RMSE, R2).
    """
    # Ensure predictions, targets, and class_indices are of the same shape
    assert predictions.shape == targets.shape == class_indices.shape, "Shape mismatch!"

    # Get the unique classes in class_indices
    unique_classes = np.unique(class_indices)

    # Initialize a dictionary to hold metrics for each class
    class_metrics = {}

    # Iterate through each unique class
    for cls in unique_classes:
        # Create a mask for the current class
        mask = class_indices == cls

        # Apply the mask to filter out the elements corresponding to the current class
        class_predictions = predictions[mask]
        class_targets = targets[mask]

        # Skip the class if there are no elements (e.g., if a class is absent)
        if class_predictions.size == 0:
            continue

        # Calculate the metrics for this class
        mse = np.mean((class_predictions - class_targets) ** 2)
        mae = np.mean(np.abs(class_predictions - class_targets))
        rmse = np.sqrt(mse)

        # R2 (Coefficient of Determination)
        mean_target = np.mean(class_targets)
        total_variance = np.sum((class_targets - mean_target) ** 2)
        residual_variance = np.sum((class_targets - class_predictions) ** 2)
        r2 = 1 - (residual_variance / total_variance) if total_variance != 0 else 0.0
        epsilon = 1e-8  # To prevent division by zero
        mape = np.mean(np.abs((class_targets - class_predictions) / (class_targets + epsilon))) * 100

        # Store the metrics for this class in the dictionary
        class_metrics[int(cls)] = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        }

    return class_metrics


class TensorboardHelper:
    def __init__(self, use_tensorboard, log_dir, comment, use_labels, loss_func, learning_rate):
        self.log_dir = log_dir
        self.comment = comment
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.use_labels = use_labels
        self.writer = None

        if use_tensorboard:
            self.init()

    def init(self):
        tb_comment = f"{self.loss_func}_{self.learning_rate}"
        if self.use_labels:
            tb_comment = f"{tb_comment}_label"
        if self.comment:
            tb_comment += f"_{self.comment}"
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir, comment=tb_comment)
        print(f"Logging tensorboard summaries in path {self.log_dir}")

    def add_histogram(self, tag, values, global_step=None, bins="tensorflow", walltime=None, max_bins=None):
        if self.writer is not None:
            self.writer.add_histogram(tag, values, global_step=global_step, bins=bins, walltime=walltime, max_bins=max_bins)

    def add_histogram_if(self, condition: bool, tag, values, global_step=None, bins="tensorflow", walltime=None, max_bins=None):
        if condition:
            self.add_histogram(tag, values, global_step=global_step, bins=bins, walltime=walltime, max_bins=max_bins)

    def add_histograms(self, tags: list, values: list, global_step=None, bins="tensorflow", walltime=None, max_bins=None):
        if self.writer is not None:
            if len(tags) != len(values):
                return
            for tag, value in zip(tags, values):
                self.writer.add_histogram(tag, value, global_step=global_step, bins=bins, walltime=walltime)
            self.writer.flush()

    def add_histograms_if(self, condition: bool, tags: list, values: list, global_step=None, bins="tensorflow", walltime=None,
                       max_bins=None):
        if condition:
            self.add_histograms(tags, values, global_step=global_step, bins=bins, walltime=walltime, max_bins=max_bins)

    def add_scalar(self, *args, **kwargs):
        if self.writer is not None:
            self.writer.add_scalar(*args, **kwargs)

    def flush(self):
        self.writer.flush()


def print_var_info(var, name):
    if type(var) == torch.Tensor:
        print(f"{name}: {type(var)}, {var.dtype}, {var.shape}")
    elif type(var) == np.ndarray:
        print(f"{name}: {type(var)}, {var.dtype}, {var.shape}")
    elif type(var) == list:
        print(f"{name}: list of {len(var)} elements of type {type(var[0])}, {var[0].shape}")
    else:
        print(f"{name}: {type(var)}")