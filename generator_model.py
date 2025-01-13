import torch
from torch import nn
from residual_layers import ResidualBlock

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
    def __init__(self, in_channels=3, inter_images = 1, features=64, out_channels = None):

        super().__init__()
        self.inter_images = inter_images

        if inter_images > 0:
            self.initial = nn.Sequential(
                nn.Conv2d((in_channels*inter_images)*2, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2)
            )  
        else:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.ReLU()
            )
            inter_images = 1
            
        self.down1 = Block(features, features*inter_images, down=True, act="relu", use_dropout=False) # 64
        self.res1 = ResidualBlock(features*inter_images)
        self.down2 = Block(features*inter_images, features*4, down=True, act="relu", use_dropout=False) # 32
        
        self.res2 = ResidualBlock(features*4)
        self.down3 = Block(features*4, features*8, down=True, act="relu", use_dropout=False) # 16
        self.res3 = ResidualBlock(features*8)
        self.down4 = Block(features*8, features*8, down=True, act="relu", use_dropout=False) # 8
        self.res4 = ResidualBlock(features*8)
        self.down5 = Block(features*8, features*8, down=True, act="relu", use_dropout=False) # 4
        self.res5 = ResidualBlock(features*8)
        self.down6 = Block(features*8, features*8, down=True, act="relu", use_dropout=False) # 2
        self.res6 = ResidualBlock(features*8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU() # 1
        )

        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=False) # 2
        self.res7 = ResidualBlock(features*8)
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 4
        self.res8 = ResidualBlock(features*8)
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 8
        self.res9 = ResidualBlock(features*8)
        self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False) # 16
        self.res10 = ResidualBlock(features*8)
        self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=False) # 32
        self.res11 = ResidualBlock(features*4)
        self.up6 = Block(features*4*2, features*inter_images, down=False, act="relu", use_dropout=False) # 64
        self.res12 = ResidualBlock(features*inter_images)
        self.up7 = Block(features*inter_images*2, features, down=False, act="relu", use_dropout=False) # 128
        self.res13 = ResidualBlock(features)

        if out_channels == None: out_channels = in_channels

        self.finalup = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  # Output: (out_channels, H*2, W*2)

    
    def forward(self, x: torch.Tensor, **z):

        if self.inter_images > 0:
            assert len(z) == self.inter_images, f"Number of inter_images was {self.inter_images}, but recieved {len(z)}"
            z_concat = torch.cat([v for v in z.values()], dim=1)
            x_repeat = x.repeat(1, z_concat.shape[1]//x.shape[1], 1, 1)
            input_tensor = torch.cat([x_repeat, z_concat], dim=1)
            input_tensor = torch.cat([x_repeat,z_concat], 1)
        else: input_tensor = x

        d1 = self.initial(input_tensor)
        
        d2 = self.down1(d1)

        r1 = self.res1(d2)
        d3 = self.down2(r1)

        r2 = self.res2(d3)
        d4 = self.down3(r2)

        r3 = self.res3(d4)
        d5 = self.down4(r3)

        r4 = self.res4(d5)
        d6 = self.down5(r4)

        r5 = self.res5(d6)
        d7 = self.down6(r5)

        r6 = self.res6(d7)
        bottleneck = self.bottleneck(r6)

        up1 = self.up1(bottleneck)


        r7 = self.res7(up1)
        up2 = self.up2(torch.cat([r7, d7], 1))

        r8 = self.res8(up2)
        up3 = self.up3(torch.cat([r8, d6], 1))

        r9 = self.res9(up3)
        up4 = self.up4(torch.cat([r9, d5], 1))

        r10 = self.res10(up4)
        up5 = self.up5(torch.cat([r10, d4], 1))

        r11 = self.res11(up5)
        up6 = self.up6(torch.cat([r11, d3], 1))

        r12 = self.res12(up6)
        up7 = self.up7(torch.cat([r12, d2], 1))

        r13 = self.res13(up7)

        # print("r13: ", r13.shape)
        # print("d1: ", d1.shape)

        return self.finalup(torch.cat([r13, d1], 1))

## Testing
if __name__ == "__main__":
    gen = Generator(in_channels=3, inter_images=0, out_channels=3)
    a = 256
    x = torch.rand(2, 3, a, a)
    result = gen(x, z1=x)
    print("Output: ", result.shape)

    total_params = sum(p.numel() for p in gen.parameters())
    print(f"Number of parameters: {total_params}")