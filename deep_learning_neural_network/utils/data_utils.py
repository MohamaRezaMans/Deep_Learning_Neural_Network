import os
import torch
from torch import Tensor
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader
from typing import Tuple
from datetime import datetime
import copy


from deep_learning_neural_network import DEEP_LEARNING_NEURAL_NETWORK_ROOT_DIR


def get_log_dir(log_dir, train_label: str) -> str:
    """
    Create or resolve the logging directory for a training run.

    If log_dir == -1, a timestamped directory is automatically created:
    logs/train_label/timestamp/

    """

    if log_dir == -1:
        # directory to save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_")
        log_root = os.path.join(DEEP_LEARNING_NEURAL_NETWORK_ROOT_DIR, "logs")
        log_dir = os.path.join(log_root, train_label, timestamp)
        os.makedirs(log_dir, exist_ok=True)

    try:
        # check if path exists as a file
        if os.path.isfile(log_dir):
            raise ValueError(f"{log_dir} exists and is a file, expected a directory.")

        os.makedirs(log_dir, exist_ok=True)

    except OSError as e:
        raise RuntimeError(f"Could not create or access log directory: {log_dir}") from e

    return log_dir

def split_dataset(
    X: Tensor,
    Y: Tensor,
    val_split: float = 0.2,
    test_split: float = 0.0,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split tensors into training, validation and test sets

    Args:
        X (Tensor): Feature tensor of shape (N, D)
        Y (Tensor): Target tensor of shape (N, 1) or (N,0)
        val_split (float): Fraction of the dataset used for validation
        test_split (float): Fraction of the dataset used for test


    Returns:
        train_dataset (Dataset): Training Dataset
        val_dataset (Dataset): Validation Dataset
        test_dataset (Dataset): Test Dataset
    """
    if val_split + test_split >= 1:
        raise ValueError(
            "val_split + test_split must be smaller than 1"
    )

    ds = TensorDataset(X, Y) # splited dataset

    len_val = int(len(ds) * val_split)
    len_test = int(len(ds) * test_split)
    len_train = len(ds) - len_val - len_test

    train_ds, val_ds, test_ds = random_split(ds, [len_train, len_val, len_test])

    return train_ds, val_ds, test_ds


def get_dataloader(
        train_ds: Dataset,
        val_ds: Dataset,
        batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects for training and validation Datasets.

    Args:
        train_ds (Dataset): Training Dataset
        val_ds (Dataset): Validation Dataset
        batch_size (int): Number of samples per batch

    Returns:
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
    """

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl

def compute_normalization_stats(train_ds: Dataset) -> Tuple[Tensor, Tensor]:
    """
    Compute feature-wise mean and standard deviation using ONLY the training data.

    Args:
        train_dataset (Dataset): Training Dataset returned by random_split

    Returns:
        mean (Tensor): Feature-wise mean of shape (D,)
        std (Tensor): Feature-wise standard deviation of shape (D,)
    """
    X = train_ds.dataset.tensors[0]   # feature tensor
    train_indices = train_ds.indices
    train_X = X[train_indices]

    mean = train_X.mean(dim=0)
    std = train_X.std(dim=0)
    std = torch.clamp(std, min=1e-8)

    return mean, std
