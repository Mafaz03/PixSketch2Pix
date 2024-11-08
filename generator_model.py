import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect") if down else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        X = self.conv(x)
        return self.dropout(X) if self.use_dropout else X
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        ) # 128

        self.down1 = Block(features, features*2, down=True, act="relu", use_dropout=False) # 64
        self.down2 = Block(features*2, features*4, down=True, act="relu", use_dropout=False) # 32
        self.down3 = Block(features*4, features*8, down=True, act="relu", use_dropout=False) # 16
        self.down4 = Block(features*8, features*8, down=True, act="relu", use_dropout=False) # 8
        self.down5 = Block(features*8, features*8, down=True, act="relu", use_dropout=False) # 4
        self.down6 = Block(features*8, features*8, down=True, act="relu", use_dropout=False) # 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU() # 1
        )

        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=False) # 2
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 4
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 8
        self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 16
        self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=False) # 32
        self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=False) # 64
        self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=False) # 128

        self.finalup = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x, z):
        d1 = self.initial(torch.cat([x,z], 1))
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.finalup(torch.cat([up7, d1], 1))

## Testing
if __name__ == "__main__":
    gen = Generator()
    x = torch.rand(2, 3, 512, 512)
    z = torch.rand(2, 3, 512, 512)
    result = gen(x, z)
    print(result.shape)