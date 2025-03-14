import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=3, num_layers=4, base_channels=64):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        channels = in_channels
        for i in range(num_layers):
            self.encoder.append(DoubleConv(channels, base_channels * (2 ** i)))
            channels = base_channels * (2 ** i)

        # Middle
        self.middle = DoubleConv(channels, channels * 2)

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            self.decoder.append(
                nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(channels * 2, channels))
            channels //= 2

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder forward
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
            x = nn.MaxPool2d(2)(x)

        # Middle forward
        x = self.middle(x)

        # Decoder forward
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            enc_out = encoder_outputs[-(i//2 + 1)]
            x = torch.cat((x, enc_out), dim=1)
            x = self.decoder[i+1](x)

        return self.out_conv(x)
