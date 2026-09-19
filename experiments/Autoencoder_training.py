import os
import json
from typing import Tuple

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from deep_learning_neural_network.models import MLPNetwork, AutoencoderNetwork
from deep_learning_neural_network.configs import AutoencoderConfig
from deep_learning_neural_network.utils import get_args, set_seed, update_cfg_from_args, class_to_dict
from deep_learning_neural_network.utils import get_dataloader, get_log_dir, save_model_jit, count_trainable_params, compute_normalization_stats_dataloader
from deep_learning_neural_network.utils import get_loss, get_optimizer
from deep_learning_neural_network.pipeline import Trainer
from deep_learning_neural_network import DEEP_LEARNING_NEURAL_NETWORK_RESOURCES_DIR

class MNISTDataset(Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __getitem__(self, idx):
        x, _ = self.base_ds[idx]
        return x, x

    def __len__(self):
        return len(self.base_ds)

class LatentDataset(Dataset):
    def __init__(self, z, y):
        self.z = z
        self.y = y

    def __getitem__(self, idx):
        return self.z[idx], self.y[idx]

    def __len__(self):
        return len(self.z)

def build_latent_dataset(model, dataloader, device):
    model.eval()
    zs = []
    ys = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            z = model.encode(xb)
            zs.append(z.cpu())
            ys.append(yb.cpu())

    zs = torch.cat(zs, dim=0)
    ys = torch.cat(ys, dim=0)
    return LatentDataset(zs, ys)

def load_dataset(root_path: str)-> Tuple[Dataset, Dataset]:
    """
    Loads the MNIST dataset and applies transformations:
    - Converts PIL images to tensors (values in [0, 1])
    - Flattens each image into a 784-dimensional vector
    """

    # transform = transforms.PILToTensor() # without normalization
    transform = transforms.Compose([
        transforms.ToTensor(), # -> [0, 1]
        transforms.Lambda(lambda x: x.view(-1)) # [28*28]T -> 784
    ])
    train_ds = datasets.MNIST(
        root=root_path,
        train=True,
        download=False,
        transform=transform
    )

    val_ds = datasets.MNIST(
        root=root_path,
        train=False,
        download=False,
        transform=transform
    )

    return train_ds, val_ds

if __name__ == "__main__":
    # get args
    args = get_args()

    # final configuration
    cfg = update_cfg_from_args(args, AutoencoderConfig())
    cfg_dict = class_to_dict(cfg)
    print(json.dumps(cfg_dict, indent=4))

    # set seed
    cfg.seed = set_seed(cfg.seed)

    # get logging directory
    cfg.logger.log_dir = get_log_dir(cfg.logger.log_dir, cfg.logger.train_label)

    # load data
    image_folder = os.path.join(DEEP_LEARNING_NEURAL_NETWORK_RESOURCES_DIR, "data")
    train_ds, val_ds = load_dataset(image_folder)

    MNIST_train_ds = MNISTDataset(train_ds)
    MNIST_val_ds = MNISTDataset(val_ds)

    # get data loaders
    # train_dl, val_dl = get_dataloader(train_ds, val_ds, cfg.training.batch_size)
    MNIST_train_dl, MNIST_val_dl = get_dataloader(MNIST_train_ds, MNIST_val_ds, cfg.training.autoencoder_batch_size)

    # load model
    autoencoder_model = AutoencoderNetwork(
        num_inputs=MNIST_train_ds[0][0].numel(),
        latent_dim=cfg.training.latent_size,
        encoder_hidden_dim=cfg.training.encoder_hidden_dims,
        decoder_hidden_dim=cfg.training.decoder_hidden_dims,
        activation=cfg.training.activation,
    )

    # loss function
    loss_fn = get_loss(cfg.training.autoencoder_loss, reduction="mean")
    # Optimizer
    optimizer = get_optimizer(cfg.training.optimizer, autoencoder_model.parameters(),
                              lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=80,
        min_lr=1e-4,
    )

    trainer = Trainer(
        autoencoder_model,
        MNIST_train_dl,
        MNIST_val_dl,
        optimizer,
        loss_fn,
        trainer_name=cfg.training.trainer.autoencoder_trainer_name,
        epochs=cfg.training.epochs,
        scheduler=scheduler,
        device=cfg.device,
        noise_std=cfg.training.trainer.noise.autoencoder_noise_std,
        noise_frac=cfg.training.trainer.noise.autoencoder_noise_frac,
        metrics=cfg.training.trainer.autoencoder_metrics,
        monitor=cfg.training.trainer.autoencoder_monitor,
        mode=cfg.training.trainer.autoencoder_mode,
        early_stopping=cfg.training.trainer.early_stopping,
        patience=cfg.training.trainer.autoencoder_patience,
        enable_plots=cfg.training.trainer.enable_plots,
        log_dir=cfg.logger.log_dir
    )

    print("Starting autoencoder training...")
    best_model = trainer.train() # Note: best_model is stored on CPU for portability

    # save model as a jit file
    autoencoder_model_path = save_model_jit(best_model, cfg.logger.log_dir, cfg.logger.autoencoder_save_model_label)

    # save config
    config_path = os.path.join(cfg.logger.log_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=4)

    print(f"config saved to {config_path}")
