from deep_learning_neural_network.configs import BaseConfig


class CNNConfig(BaseConfig):
    seed = -1  # different random numbers

    class Training(BaseConfig.Training):
        mlp_hidden_dims = [512]
        batch_size = 512
        epochs = 150
        learning_rate = 1e-4

        mlp_activation = 'relu'
        cnn_activation = 'relu'
        optimizer = "adam"
        weight_decay = 1e-4
        loss = "crossentropy"  # loss calculation function

        conv_blocks = [
            {
                "out_channels": 32,
                "num_convs": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": {"size": 2, "stride": 1, "padding": 0},
                "batch_normalization": True
            },
            {
                "out_channels": 64,
                "num_convs": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": {"size": 2, "stride": 2, "padding": 0},
                "batch_normalization": True
            },
            {
                "out_channels": 128,
                "num_convs": 2,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": {"size": 2, "stride": 2, "padding": 0},
                "batch_normalization": True
            },
        ]

        class Trainer(BaseConfig.Training.Trainer):
            trainer_name = "CNN"
            matrics = ["accuracy"]
            monitor = "accuracy"
            mode = "max"
            enable_plots = True
            early_stopping = True
            patience = 15

            add_noise = True

            class Noise(BaseConfig.Training.Trainer.Noise):
                noise_std = 0.0
                noise_frac = 0.0

    class DataGeneration:
        num_classes = 10
        num_channels = 1
        samples_per_class = 2000
        image_dim = 14
        mean_range = [-5, 4]
        variance = 2.25

    class Evaluation(BaseConfig.Evaluation):
        val_split = 0.2  # Proportion of training data to use for validation
        load_run = -1

    class Logger(BaseConfig.Logger):
        train_label = 'CNN'
        save_model_label = "CNN_JIT_model"
        log_dir = -1

        experiment = "cnn_training"
        dataset = "Toy_Data"


