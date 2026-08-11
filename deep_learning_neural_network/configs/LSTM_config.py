from deep_learning_neural_network.configs import BaseConfig


class LSTMConfig(BaseConfig):

    class Training(BaseConfig.Training):
        hidden_dims = [16, 32]
        epochs = 30
        batch_size = 128
        learning_rate = 1e-3

        activation = 'relu'
        optimizer = "adam"
        weight_decay = 1e-4
        loss = "mse"

        class Trainer(BaseConfig.Training.Trainer):
            trainer_name = "MLP"
            metrics = ["mse", "rmse",
                       "r2"]
            monitor = "rmse"
            mode = "min"
            enable_plots = True
            early_stopping = False
            patience = 100

            add_noise = True

            class Noise(BaseConfig.Training.Trainer.Noise):
                noise_std = 0.005
                noise_frac = 0.02

    class Evaluation(BaseConfig.Evaluation):
        val_split = 0.2
        load_run = -1

    class Logger(BaseConfig.Logger):
        train_label = 'LSTM'
        save_model_label = "LSTM_JIT_model"
        log_dir = -1

        experiment = "rnn_training"  # Name of the overall experiment
        dataset = "rt_polarity"  # Name of the dataset being used

