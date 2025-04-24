import torch
import torch.nn as nn


class Conv3DEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2,
                      padding=1),  # -> (16, 128, 120)
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2,
                      padding=1),           # -> (8, 64, 60)
            nn.ReLU(),
            nn.Conv3d(64, hidden_dim, kernel_size=3,
                      stride=2, padding=1),   # -> (4, 32, 30)
            nn.ReLU()
        )

    def forward(self, x):  # (B, 1, 32, 256, 240)
        return self.encoder(x)  # (B, hidden_dim, D', H', W')


class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, depth=2):
        super().__init__()
        self.flatten = nn.Flatten(2)  # Flatten spatial dims
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)

    def forward(self, x):  # (B, 3, C, D, H, W)
        B, T, C, D, H, W = x.shape
        x = x.view(B * T, C, D, H, W)
        x = x.view(B, T, C, -1).mean(-1)  # Global average pooling: (B, T, C)
        x += self.pos_embedding
        x = self.transformer(x.permute(1, 0, 2))  # (T, B, C)
        return x.permute(1, 0, 2)  # (B, T, C)


class Conv3DDecoderSSL(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, 64, kernel_size=4,
                               stride=2, padding=1),  # -> (8, 64, 60)
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2,
                               padding=1),           # -> (16, 128, 120)
            nn.ReLU(),
            nn.ConvTranspose3d(32, out_channels, kernel_size=4,
                               stride=2, padding=1),  # -> (32, 256, 240)
        )

    def forward(self, x):
        return self.decoder(x)  # (B, 3, 32, 256, 240)


class Conv3DDecoderFinetune(nn.Module):
    def __init__(self, out_channels=1, hidden_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, 64, kernel_size=4,
                               stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(
                32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)  # (B, 1, 32, 256, 240)


class MRISequenceSSL(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = Conv3DEncoder(hidden_dim=hidden_dim)
        self.temporal_model = TemporalTransformer(hidden_dim=hidden_dim)
        self.decoder = Conv3DDecoderSSL(hidden_dim=hidden_dim)

    def forward(self, x):  # (B, 3, 32, 256, 240)
        B, T, D, H, W = x.shape
        x = x.unsqueeze(2)  # (B, 3, 1, 32, 256, 240)
        encoded = torch.stack([self.encoder(x[:, t])
                              for t in range(T)], dim=1)  # (B, 3, C, D', H', W')
        temporal = self.temporal_model(encoded)  # (B, 3, C)

        # Decode using last timestep (or pool)
        pooled = temporal.mean(dim=1)  # (B, C)
        D_out, H_out, W_out = encoded.shape[-3:]
        recon_input = pooled.view(
            B, -1, 1, 1, 1).expand(-1, -1, D_out, H_out, W_out)
        out = self.decoder(recon_input)  # (B, 3, 32, 256, 240)
        return out


# # Input batch: 8 samples, 3 sequential scans per sample, each of shape 32x256x240
# x = torch.randn(8, 3, 32, 256, 240)
# model = MRISequenceSSL()
# y = model(x)  # y: predicted scan of shape (8, 1, 32, 256, 240)
# print(y.shape)