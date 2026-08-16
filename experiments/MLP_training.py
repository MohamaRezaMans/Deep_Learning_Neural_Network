import os
import json

import pandas as pd
import torch

from deep_learning_neural_network.models import MLPNetwork
from deep_learning_neural_network.configs import MLPConfig
from deep_learning_neural_network.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_neural_network.utils import get_log_dir, split_dataset, get_dataloader, compute_normalization_stats, save_model_jit
from deep_learning_neural_network.utils import get_loss, get_optimizer
# from deep_learning_neural_network.pipeline import Trainer
from deep_learning_neural_network import DEEP_LEARNING_NEURAL_NETWORK_RESOURCES_DIR

def load_data(data_dir: str):
    """
    Load and preprocess the Life Expectancy dataset.

    csv_path : str
        Full path to the CSV file.

    Returns
    Feature : torch.Tensor
        Feature matrix (float32).
    Target : torch.Tensor
        Target values (Life Expectancy).
    """

    df = pd.read_csv(data_dir, sep = ',') # sep = '&' -> column1&column2&...

    # Preprocessing dataset
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    df = df.drop(columns=["Country"]) # remove country column
    df["Status"] = df["Status"].map({"Developed": 1, "Developing": 0}) # Encode categorical column
    # print(df["Status"].unique())

    # Handle missing values with median
    categorical_cols = ["Status"]
    numeric_cols = df.columns.drop(categorical_cols)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df["Status"] = df["Status"].fillna(df["Status"].mode()[0])
    # print(df.isna().sum())

    # Convert to torch tensors
    features = torch.tensor(df.drop(columns=["Life_expectancy"]).values, dtype=torch.float32)
    targets = torch.tensor(df[["Life_expectancy"]].values, dtype=torch.float32)

    return features, targets

if __name__ == "__main__":
    # get args
    args = get_args()

    # final configuration
    cfg = update_cfg_from_args(args, MLPConfig())
    cfg_dict = class_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=4))

    # set seed
    cfg.seed = set_seed(cfg.seed)

    # get logging directory
    cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    # load dataset
    csv_folder = os.path.join(DEEP_LEARNING_NEURAL_NETWORK_RESOURCES_DIR, "data", "Life_Expectancy_Data")
    csv_path = os.path.join(csv_folder, "Life Expectancy Data.csv")
    X, Y = load_data(csv_path)
    print("X shape", X.size())
    print("Y shape", Y.size())

    # split
    train_ds, val_ds, _ = split_dataset(X, Y, cfg.evaluation.val_split)

    # get data loaders
    train_dl, val_dl = get_dataloader(train_ds, val_ds, cfg.training.batch_size)

    # compute_normalization_stats
    mean, std = compute_normalization_stats(train_ds)
    print("Feature mean:", mean.cpu().numpy())
    print("Feature std: ", std.cpu().numpy())
    print(f"output min: {Y.min().item():.4f}, max: {Y.max().item():.4f}, std: {Y.std().item():.4f}")

    # load model
    mlp_model = MLPNetwork(
        num_inputs=X.size(1),
        num_outputs=Y.size(1),
        network_hidden_dims=cfg.training.hidden_dims,
        activation=cfg.training.activation)

    mlp_model.set_normalization(mean, std)

    # loss function
    loss_fn = get_loss(cfg.training.loss, reduction="mean")

