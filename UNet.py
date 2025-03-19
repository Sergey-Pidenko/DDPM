import torch
import torch.nn as nn

def get_time_embedding(t, embedding_size=64):
    device = t.device
    half_dim = embedding_size // 2
    emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = t.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.adjust_channels = (in_channels != out_channels)
        if self.adjust_channels:
            self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        res = x
        if self.adjust_channels:
            res = self.channel_adjust(res)
        return self.relu(self.double_conv(x) + res)

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, num_layers=6, base_channels=64, time_embedding_size=64):
        super(UNet, self).__init__()
        
        self.time_embedding_size = time_embedding_size

        # Encoder
        self.encoder = nn.ModuleList()
        channels = in_channels
        for i in range(num_layers):
            self.encoder.append(ResidualBlock(channels + time_embedding_size, base_channels * (2 ** i)))
            channels = base_channels * (2 ** i)

        # Middle
        self.middle = nn.Sequential(
            ResidualBlock(channels + time_embedding_size, channels * 2),
            nn.Dropout(p=0.5)
        )

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers):
            self.decoder.append(
                nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
            )
            self.decoder.append(ResidualBlock(2 * channels + time_embedding_size, channels))
            channels //= 2

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        encoder_outputs = []
        t_embedding = get_time_embedding(t, self.time_embedding_size)
        t_embedding = t_embedding.view(t_embedding.size(0), self.time_embedding_size, 1, 1)
        
        for layer in self.encoder:
            t_emb_expanded = t_embedding.expand(x.size(0), -1, x.size(2), x.size(3))
            x = torch.cat([x, t_emb_expanded], dim=1)
            x = layer(x)
            encoder_outputs.append(x)
            x = nn.MaxPool2d(2)(x)

        t_emb_expanded = t_embedding.expand(x.size(0), -1, x.size(2), x.size(3))
        x = torch.cat([x, t_emb_expanded], dim=1)
        x = self.middle(x)

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            enc_out = encoder_outputs[-(i//2 + 1)]
            t_emb_expanded = t_embedding.expand(x.size(0), -1, x.size(2), x.size(3))
            
            # Конкатенируются ConvTranspose2d результат, skip connection и временной эмбеддинг,
            # что даёт 2 * channels + time_embedding_size каналов.
            x = torch.cat((x, enc_out), dim=1)
            x = torch.cat([x, t_emb_expanded], dim=1)
            
            x = self.decoder[i+1](x)

        return self.out_conv(x)