import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=4, 
                      stride=stride,
                      padding=1,
                      bias="False",
                      padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )


    def forward(self, x):
        return self.conv(x)



class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features = [64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels*2, 
                out_channels=features[0], 
                kernel_size=4, 
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_c = features[0]
        for feature in features[1:]:
            layers.append(ConvBlock(in_c, feature, stride= 1 if feature == features[-1] else 2))
            in_c = feature

        layers.append(
            nn.Conv2d(
                in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )

        # layers.append(nn.AdaptiveAvgPool2d((30, 30)))               ### try this later
        self.all = nn.Sequential(*layers)
    
    def forward(self, x, y):
        X = torch.cat([x, y], dim=1)
        # print(X.shape)
        X = self.initial(X)
        # print(X.shape)
        return self.all(X)
        
# Test
if __name__ == "__main__":
    disc = Discriminator(in_channels=3)
    x = torch.rand(5,3,256,256)
    y = torch.rand(5,3,256,256)
    output = disc(x, y)
    print(output.shape)