import os
import torch
import numpy as np
import random
import argparse

def set_seed(seed: int):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    # print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def get_args():
    parser = argparse.ArgumentParser()
    # NOTICE: All 'default=...' arguments are removed. They will default to None.
    parser.add_argument("--val_split", type=float) # Ratio of validation data to total data (0.2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_dims", type=int, nargs="+",
                        help="Hidden layer sizes, e.g. --hidden_dims 64 64")
    parser.add_argument("--add_noise", type=lambda x: x.lower() == "true",
                        help="Enable or disable input noise (true/false)")
    parser.add_argument("--noise_std", type=float, help="Absolute input noise (units of X)")
    parser.add_argument("--noise_frac", type=float, help="Fraction of per-feature std")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    return args

def update_cfg_from_args(input_args, defaults_cfg):
    if input_args.seed is not None:
        defaults_cfg.seed = input_args.seed

    if input_args.device is not None:
        defaults_cfg.device = input_args.device

    # training
    if input_args.epochs is not None:
        defaults_cfg.training.epochs = input_args.epochs

    if input_args.batch_size is not None:
        defaults_cfg.training.batch_size = input_args.batch_size

    if input_args.learning_rate is not None:
        defaults_cfg.training.learning_rate = input_args.learning_rate

    if input_args.hidden_dims is not None:
        defaults_cfg.training.hidden_dims = input_args.hidden_dims

    if input_args.add_noise is not None:
        defaults_cfg.training.trainer.add_noise = input_args.add_noise

    # noise config
    if input_args.noise_std is not None:
        defaults_cfg.training.trainer.noise.noise_std = input_args.noise_std

    if input_args.noise_frac is not None:
        defaults_cfg.training.trainer.noise.noise_frac = input_args.noise_frac

    # evaluation
    if input_args.val_split is not None:
        defaults_cfg.evaluation.val_split = input_args.val_split

    # enforce rule
    if not defaults_cfg.training.trainer.add_noise:
        defaults_cfg.training.trainer.noise.noise_std = 0.0
        defaults_cfg.training.trainer.noise.noise_frac = 0.0

    return defaults_cfg
