"""Pytorch models for material property prediction."""
import torch.nn as nn


class ElemNet(nn.Module):
    """Torch implementation of ElemNet.

    Cite the original paper for the model:
    Jha, D., Ward, L., Paul, A. et al.
    ElemNet: Deep Learning the Chemistry of Materials
    From Only Elemental Composition.
    Sci Rep 8, 17593 (2018).
    https://doi.org/10.1038/s41598-018-35934-y

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        """Initialise the model.

        Args:
            input_dim (int): Input dimensionality.
            output_dim (int): Output dimensionality.

        """
        super().__init__()

        # Layers: 1-4 have 1024 hidden units with a final dropout of 0.8
        self.fc_layers1 = MLP(input_dim, [1024, 1024, 1024], 1024)
        self.dropout1 = nn.Dropout(0.8)
        # Layers: 5-7 have 512 hidden units with a final dropout of 0.9
        self.fc_layers2 = MLP(1024, [512, 512], 512)
        self.dropout2 = nn.Dropout(0.9)
        # Layers: 8-10 have 256 hidden units with a final dropout of 0.7
        self.fc_layers3 = MLP(512, [256, 256], 256)
        self.dropout3 = nn.Dropout(0.7)
        # Layers: 11-13 have 128 hidden units with a final dropout of 0.8
        self.fc_layers4 = MLP(256, [128, 128], 128)
        self.dropout4 = nn.Dropout(0.8)
        # Layers 14-15 have 64 hidden units, layer 16 has 32 hidden units,
        #  and layer 17 has `output_dim` hidden unit(s)
        self.fc_layers5 = MLP(128, [64, 64, 32], output_dim)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        x = self.fc_layers1(x)
        x = self.dropout1(x)
        x = self.fc_layers2(x)
        x = self.dropout2(x)
        x = self.fc_layers3(x)
        x = self.dropout3(x)
        x = self.fc_layers4(x)
        x = self.dropout4(x)
        x = self.fc_layers5(x)
        x = x.squeeze()
        return x


class MLP(nn.Module):
    """Multi-layer perceptron."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        activation=nn.ReLU(),
    ):
        """Initialise the model.

        Args:
            input_size (int): Input dimensionality.
            hidden_sizes (list): List of hidden layer sizes.
            output_size (int): Output dimensionality.
            activation (nn.Module): Activation function.

        """
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(activation)

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the model.

        Args:

            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.mlp(x)
