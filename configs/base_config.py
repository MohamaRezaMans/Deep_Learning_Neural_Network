import torch
from sympy import true


class BaseConfig():
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class training:
        hidden_dims = [64, 32, 16]
        epochs = 3000
        batch_size = 64
        learning_rate = 0.0001

        activation = 'lrelu'
        optimizer = 'adam'
        weight_decay = 0.0001
        loss = "mse"
        loss_beta = 1

        class trainer():
            trainer_name = "MLP"
            metric = ["mse", "rmse"]
            monitor = "rmse"
            mode = "min"
            enable_plots = True
            early_stopping = True
            patience = 100

            add_noise = True
            class noise():
                noise_std = 0.005
                noise_frac = 0.02

    class evaluation:
        val_split = 0.2
        load_run = -1

    class logger:
        train_label = 'MLP'
        save_model_label = "JIT_model"
        log_dir = -1

        experiment = "exp"
        dataset = "Dataset"
