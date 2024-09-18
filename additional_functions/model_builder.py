import torch
from torch import nn

class TinyVGG(nn.Module):
  def __init__(self,
               input_channels: int,
               hidden_channels: int,
               output_channels: int):
    super().__init__()
    self.layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_channels,
                  out_channels=hidden_channels,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=hidden_channels,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2,
                     padding=0)
    )
    self.layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=hidden_channels,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_channels,
                  out_channels=hidden_channels,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2,
                     padding=0)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_channels * 13 * 13,
                  out_features=output_channels)
    )

  def forward(self, x):
    x = self.layer_1(x)
    #print(x.shape)
    x = self.layer_2(x)
    #print(x.shape)
    x = self.classifier(x)
    return x