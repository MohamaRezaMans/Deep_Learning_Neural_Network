import torch
import torch.nn as nn  # nn : loss functions and activation functions


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu": return nn.ELU()
    if name == "relu": return nn.ReLU()
    if name == "lrelu": return nn.LeakyReLU()
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    if name == "softsign": return nn.Softsign()
    raise ValueError(f"Unknown activation '{name}'")
