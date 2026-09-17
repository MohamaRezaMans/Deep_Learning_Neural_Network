import torch
import torch.nn as nn
from typing import List

from deep_learning_neural_network.utils import get_activation

class AutoencoderNetwork(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 latent_dim: int,
                 encoder_hidden_dim: List[int] = [128, 8],
                 decoder_hidden_dim: List[int] = [128, 8],
                 activation: str = 'relu',
                 **kwargs):

        if kwargs:
            print("AutoEncoderNetwork.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        super().__init__()

        # normalization buffers
        self.register_buffer("x_mean", torch.zeros(num_inputs))
        self.register_buffer("x_std", torch.ones(num_inputs))
        self.normalize_inputs = False

        encoder_layers = []
        if len(encoder_hidden_dim) == 0:
            encoder_layers.append(nn.Linear(num_inputs, latent_dim))
        else:
            encoder_layers.append(nn.Linear(num_inputs, encoder_hidden_dim[0]))
            encoder_layers.append(get_activation(activation))
            for i in range(len(encoder_hidden_dim)):
                if i == len(encoder_hidden_dim) - 1:
                    encoder_layers.append(nn.Linear(encoder_hidden_dim[i], latent_dim))
                    encoder_layers.append(get_activation(activation))
                else:
                    encoder_layers.append(nn.Linear(encoder_hidden_dim[i], encoder_hidden_dim[i + 1]))
                    encoder_layers.append(get_activation(activation))
        self.encoder_net = nn.Sequential(*encoder_layers)

        decoder_layers = []
        if len(decoder_hidden_dim) == 0:
            decoder_layers.append(nn.Linear(latent_dim, num_inputs))
        else:
            decoder_layers.append(nn.Linear(latent_dim, decoder_hidden_dim[0]))
            decoder_layers.append(get_activation(activation))
            for i in range(len(decoder_hidden_dim)):
                if i == len(decoder_hidden_dim) - 1:
                    decoder_layers.append(nn.Linear(decoder_hidden_dim[i], num_inputs))
                    decoder_layers.append(get_activation(activation))
                else:
                    decoder_layers.append(nn.Linear(decoder_hidden_dim[i], decoder_hidden_dim[i + 1]))
                    decoder_layers.append(get_activation(activation))
        self.decoder_net = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       z = self.encode(x)
       x_ret = self.decode(z)
       return  x_ret

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_inputs:
            x = (x - self.x_mean) / self.x_std
        return self.encoder_net(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_ret = self.decoder_net(z)
        if self.normalize_inputs:
            x_ret =  x_ret * self.x_std + self.x_mean
        return x_ret

    @torch.no_grad()
    def set_normalization(self, mean, std, eps: float = 1e-6):
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        std_t = torch.as_tensor(std, dtype=self.x_std.dtype, device=self.x_std.device).clamp_min(eps)

        self.x_mean.copy_(mean_t)
        self.x_std.copy_(std_t)
        self.normalize_inputs = True