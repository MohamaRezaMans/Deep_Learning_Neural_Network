import torch
import torch.nn as nn
from typing import List

from deep_learning_neural_network.utils import get_activation


class MLPNetwork(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 network_hidden_dims: List[int] = [32, 32, 32],
                 activation: str = 'softsign',
                 **kwargs):  # extra arguments
        if kwargs:
            print("MLPNetwork.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))

        super().__init__()

        # normalization buffers
        self.register_buffer("x_mean", torch.zeros(num_inputs))
        self.register_buffer("x_std", torch.ones(num_inputs))
        self.normalize_inputs = False

        mlp_input_dim = num_inputs
        mlp_output_dim = num_outputs

        # network
        net_layers = []
        if len(network_hidden_dims) == 0:
            net_layers.append(nn.Linear(mlp_input_dim, mlp_output_dim))

        else:
            net_layers.append(nn.Linear(mlp_input_dim, network_hidden_dims[0]))
            net_layers.append(get_activation(activation))
            for l in range(len(network_hidden_dims)):
                if l == len(network_hidden_dims) - 1:
                    net_layers.append(nn.Linear(network_hidden_dims[l], mlp_output_dim))
                else:
                    net_layers.append(nn.Linear(network_hidden_dims[l], network_hidden_dims[l + 1]))
                    net_layers.append(get_activation(activation))
        self.mlp_net = nn.Sequential(*net_layers)

        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.mlp_net[-1].weight, gain=0.01)
        nn.init.zeros_(self.mlp_net[-1].bias)

        # print(f"MLP Network Structure: {self.mlp_net}")

    def features(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.mlp_net):
            if index < len(self.mlp_net) - 1:
                x = layer(x)
        return x

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp_net[-1](features)

    def classifier_weight(self) -> torch.Tensor:
        return self.mlp_net[-1].weight

    @torch.no_grad()
    def set_normalization(self, mean, std, eps: float = 1e-6):
        """Store per-feature mean/std (from TRAIN split). Enables normalization in forward()."""
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        std_t = torch.as_tensor(std, dtype=self.x_std.dtype, device=self.x_std.device).clamp_min(eps)
        self.x_mean.copy_(mean_t)
        self.x_std.copy_(std_t)
        self.normalize_inputs = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            x = (x - self.x_mean) / self.x_std
        return self.mlp_net(x)


def _orthogonal_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


obj = MLPNetwork(5, 5)
print(obj.forward(torch.randn(1, 5)))