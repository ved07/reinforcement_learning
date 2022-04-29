import torch
import torch.nn as nn

class DeepQ(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels= input_dim, out_channels= 16, kernel_size=8, stride=4),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=2),
            nn.Conv2d(in_channels=32, out_channels= 32, kernel_size=7, stride=2),
            nn.Flatten(),
            nn.Linear(1728, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        return self.model(x)
