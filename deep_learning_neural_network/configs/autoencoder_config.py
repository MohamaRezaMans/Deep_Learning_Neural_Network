from deep_learning_neural_network.configs import BaseConfig


class AutoencoderConfig(BaseConfig):

    class Training(BaseConfig.Training):
        encoder_hidden_dims = [128] # number of neurons for encoder
        decoder_hidden_dims = [128] # number of neurons for decoder
        classifier_hidden_dims = [16]  # number of neurons for classifier
        latent_size =  32 # 32 bit

        classifier_num_outputs = 10 # number of outputs class
        autoencoder_batch_size = 256 # the number of input data to autoencoder
        classifier_batch_size = 512 # the number of input data to classifier
        learning_rate = 1e-3 # + - rate
        epochs = 30

        activation = 'relu'
        optimizer = "adam"
        weight_decay = 1e-5  # L2 regularization, This extra term penalizes large weights.
        autoencoder_loss = "mse"
        classifier_loss = "crossentropy"

        class Trainer(BaseConfig.Training.Trainer):
            autoencoder_trainer_name = "Autoencoder"
            classifier_trainer_name = "Classifier"
            autoencoder_metrics = ["mse", "rmse", "r2"]
            autoencoder_monitor = "rmse"
            autoencoder_mode = "min"
            classifier_metrics = ["accuracy"]
            classifier_monitor = "accuracy"
            classifier_mode = "max"
            enable_plots = True # save plots of metrics
            early_stopping = True
            autoencoder_patience = 5 # Number of epochs with no improvement before stopping
            classifier_patience = 25

            add_noise = True

            class Noise(BaseConfig.Training.Trainer.Noise):
                autoencoder_noise_std = 0.005
                autoencoder_noise_frac = 0.02
                classifier_noise_std = 0
                classifier_noise_frac = 0

    class Evaluation(BaseConfig.Evaluation):
        load_run = -1
        load_autoencoder = -1

    class Logger(BaseConfig.Logger):
        train_label = 'Autoencoder'
        autoencoder_save_model_label = "Autoencoder_JIT_model"
        classifier_save_model_label = "Classifier_JIT_model"
        log_dir = -1 # Directory to save logs (-1 means use default location)

        experiment = "autoencoder_training"  # Name of the overall experiment
        dataset = "MNIST"  # Name of the dataset being used
