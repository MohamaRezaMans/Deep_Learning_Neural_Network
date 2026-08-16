import torch



class BaseConfig:
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class training:
        hidden_dims = [16, 32]
        epochs = 3000 # number of step
        batch_size = 64
        learning_rate = 0.0001

        activation = 'lrelu' # Makes the network activation function nonlinear.
        optimizer = 'adam' # Optimum algorithm: Calculation of weight changes is based on the gradient.
        weight_decay = 0.0001
        loss = "mse" # loss calculation function
        #loss_beta = 1

        class trainer:
            trainer_name = "MLP"
            metric = ["mse", "rmse"]
            monitor = "rmse"
            mode = "min"
            enable_plots = True
            early_stopping = True # Stop if there is no improvement.
            patience = 100 # Number of epochs with no improvement before stopping

            add_noise = True
            class noise:
                noise_std = 0.005 # Standard deviation of the Gaussian noise to add
                noise_frac = 0.02 # Fraction of samples that will receive noise perturbation

    class evaluation:
        val_split = 0.2
        load_run = -1 # -1 means the last run, load

    class logger:
        train_label = 'MLP'
        save_model_label = "JIT_model"
        log_dir = -1

        experiment = "exp"
        dataset = "Dataset"
