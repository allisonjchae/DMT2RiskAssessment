"""
Baseline Fully Connected Neural Network (FCNN) implementation.

Author(s):
    Allison Chae
    Michael Yao

Licensed under the MIT License.
"""
from collections import OrderedDict
import torch
from torch import nn


class FCNN(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 128,
        num_layers: int = 8,
        activation: str = "ReLU"
    ):
        """
        Args:
            in_chans: number of input channels for the first layer.
            out_chans: number of output channels from the last layer.
            chans: number of channels in the intermediate layers.
            num_layers: total number of layers.
            activation: activation function. One of [`Sigmoid`, `Softplus`,
                `ReLU`, `LeakyReLU`].
        """
        super().__init__()
        self.layers = [
            ("linear0", nn.Linear(in_chans, chans),),
        ]
        self.activation = activation.lower()
        if self.activation == "sigmoid":
            self.layers.append(("activation0", nn.Sigmoid()))
        elif self.activation == "softplus":
            self.layers.append((
                "activation0", nn.Softplus(beta=1, threshold=20)
            ))
        elif self.activation == "relu":
            self.layers.append(("activation0", nn.ReLU()))
        elif self.activation == "leakyrelu":
            self.layers.append(("activation0", nn.LeakyReLU()))
        else:
            raise ValueError(f"Unrecognized activation function {activation}.")

        for i in range(1, num_layers - 1):
            self.layers.append(("linear" + str(i), nn.Linear(chans, chans)))
            if self.activation == "sigmoid":
                self.layers.append(("activation" + str(i), nn.Sigmoid()))
            elif self.activation == "softplus":
                self.layers.append((
                    "activation" + str(i), nn.Softplus(beta=1, threshold=20)
                ))
            elif self.activation == "relu":
                self.layers.append(("activation" + str(i), nn.ReLU()))
            elif self.activation == "leakyrelu":
                self.layers.append(("activation" + str(i), nn.LeakyReLU()))

        self.layers.append(
            ("linear" + str(num_layers - 1), nn.Linear(chans, out_chans))
        )

        self.layers = nn.Sequential(OrderedDict(self.layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multilayer perceptron.
        Input:
            x: input feature tensor with a size of in_chans.
        Return:
            FCNN(x).
        """
        return self.layers(x)
