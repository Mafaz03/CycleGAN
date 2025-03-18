import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down: bool = True, use_act: bool = True, **kwargs):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),           
            nn.ReLU(inplace=True) if use_act else nn.Identity() # no ops
        )
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, use_act=True, down=True, kernel_size = 3, padding = 1),
            ConvBlock(in_channels=channels, out_channels=channels, use_act=False, down=True, kernel_size = 3, padding = 1),
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, img_channels, num_features = 64, num_residuals=9):
        super().__init__()

        self.initial = nn.Sequential(
            ConvBlock(img_channels, num_features, kernel_size=7, stride = 1, padding = 3),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, down=True, use_act=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, down=True, use_act=True, kernel_size=3, stride=2, padding=1)
            ]
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )

        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, use_act=True, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, use_act=True, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.final = nn.Conv2d(in_channels=num_features, out_channels=img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.final(x))
    
# Testing  
if __name__ == "__main__":
    x = torch.rand(5, 3, 256, 256)
    model = Generator(img_channels=3)
    output = model(x)
    print(output.shape)