import torch
import torch.nn as nn  # nn : loss functions and activation functions
from deep_learning_neural_network.utils import IAMLoss
from deep_learning_neural_network.utils import AngularMarginLoss

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu": return nn.ELU()
    if name == "relu": return nn.ReLU()
    if name == "lrelu": return nn.LeakyReLU()
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    if name == "softsign": return nn.Softsign()
    raise ValueError(f"Unknown activation '{name}'")

def get_loss(name: str, **kwargs) -> nn.Module:
    '''
    Map a string name to a PyTorch loss module.
    Extra keyword args (in kwargs) are forwarded to the loss constructor.
    '''
    name = name.lower()
    if name == "mse": return nn.MSELoss(**kwargs)
    if name == "smooth_l1": return nn.SmoothL1Loss(**kwargs)
    if name == "l1": return nn.L1Loss(**kwargs)
    if name == "crossentropy": return nn.CrossEntropyLoss(**kwargs)
    if name == "iam": return IAMLoss(**kwargs)
    if name == "angularmargin": return AngularMarginLoss(**kwargs)
    raise ValueError(f"Unknown loss '{name}'")

