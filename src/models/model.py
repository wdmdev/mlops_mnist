import torch
from torch import nn


class MyNeuralNet(nn.Module):
    """Basic neural network class.

    Args:
        in_features: number of input features
        out_features: number of output features

    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x
