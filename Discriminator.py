import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, 
                      kernel_size=4,
                      stride=stride,
                      bias=True, 
                      padding=1,
                      padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, features = [64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
            in_channels=in_channels,
            out_channels=features[0],
            kernel_size=4,
            stride=2,
            padding=1,
            padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels=in_channels, out_channels=feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, padding=1, stride=1, padding_mode="reflect")
        )
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return torch.sigmoid(self.model(self.initial(x)))
    
# Testing  
if __name__ == "__main__":
    x = torch.rand(5, 3, 256, 256)
    model = Discriminator(in_channels=3)
    output = model(x)
    print(output.shape)