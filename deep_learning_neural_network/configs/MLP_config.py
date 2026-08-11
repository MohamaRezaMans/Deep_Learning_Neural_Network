from deep_learning_neural_network.configs import BaseConfig


class MLPConfig(BaseConfig):
    seed = 42  # Random seed for reproducibility
    device = 'cuda'

    class Training(BaseConfig.Training):
        hidden_dims = [16, 32]
        epochs = 30
        batch_size = 128
        learning_rate = 1e-3

        activation = 'relu'
        optimizer = "adam"
        weight_decay = 1e-4
        loss = "mse"

        class trainer(BaseConfig.training.trainer):
            trainer_name = "MLP"
            metrics = ["mse", "rmse", "r2"]
            monitor = "rmse"
            mode = "min"
            enable_plots = True
            early_stopping = False
            patience = 100

            add_noise = True

            class noise(BaseConfig.training.trainer.noise):
                noise_std = 0.005
                noise_frac = 0.02

    class evaluation(BaseConfig.evaluation):
        val_split = 0.2
        load_run = -1

    class logger(BaseConfig.logger):
        train_label = 'MLP'
        save_model_label = "MLP_JIT_model"
        log_dir = -1

        experiment = "mlp_training"
        dataset = "Life_Expectancy_Data"