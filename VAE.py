import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpNoSkip, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class VAEUNet(nn.Module):
    def __init__(self, latent_dim=1024, bilinear=True):
        """
        latent_dim – размерность латентного вектора.
        Входное изображение: (3, 128, 128), выходное изображение: (3, 512, 512)
        """
        super(VAEUNet, self).__init__()
        self.latent_dim = latent_dim
        self.bilinear = bilinear
        
        # Энкодер (Contracting Path)
        self.inc = DoubleConv(3, 64)           # 64 x 128 x 128
        self.down1 = Down(64, 128)             # 128 x 64 x 64
        self.down2 = Down(128, 256)            # 256 x 32 x 32
        self.down3 = Down(256, 512)            # 512 x 16 x 16
        self.down4 = Down(512, 512)            # 512 x 8 x 8


        self.flatten_dim = 512 * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        
        # Декодер с U-Net скип-соединениями
        self.up1 = Up(512 + 512, 512, bilinear)   # 512+512 -> 512 x 16 x 16
        self.up2 = Up(512 + 256, 256, bilinear)     # 512 -> 256 x 32 x 32
        self.up3 = Up(256 + 128, 128, bilinear)     # 256 -> 128 x 64 x 64
        self.up4 = Up(128 + 64, 64, bilinear)       # 128 -> 64 x 128 x 128
        
        # Повышение разрешения до 512x512 без скип-соединений:
        self.up5 = UpNoSkip(64, 64, bilinear)       # 128 -> 256
        self.up6 = UpNoSkip(64, 64, bilinear)       # 256 -> 512
        
        self.outc = nn.Conv2d(64, 3, kernel_size=1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        x1 = self.inc(x)       # 64 x 128 x 128
        x2 = self.down1(x1)    # 128 x 64 x 64
        x3 = self.down2(x2)    # 256 x 32 x 32
        x4 = self.down3(x3)    # 512 x 16 x 16
        x5 = self.down4(x4)    # 512 x 8 x 8
        
        batch_size = x.size(0)
        x5_flat = x5.view(batch_size, -1)
        mu = self.fc_mu(x5_flat)
        logvar = self.fc_logvar(x5_flat)
        z = self.reparameterize(mu, logvar)
        
        x_decoded = self.fc_decode(z)
        x_decoded = x_decoded.view(batch_size, 512, 8, 8)
        
        x = self.up1(x_decoded, x4)   # 512 x 16 x 16
        x = self.up2(x, x3)           # 256 x 32 x 32
        x = self.up3(x, x2)           # 128 x 64 x 64
        x = self.up4(x, x1)           # 64 x 128 x 128
        
        x = self.up5(x)               # 64 x 256 x 256
        x = self.up6(x)               # 64 x 512 x 512
        
        output = self.outc(x)         # 3 x 512 x 512
        return output, mu, logvar
