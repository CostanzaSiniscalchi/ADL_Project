import torch.nn as nn
from vit_pytorch import ViT


class ScanOrderViT(nn.Module):
    def __init__(self, image_size=224, dim=256, depth=4, heads=4, mlp_dim=512):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=16,
            num_classes=6,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=1
        )
        self.encoder = self.vit
        self.encoder.mlp_head = nn.Identity()
        self.classifier = nn.Linear(dim, 6)

    def forward(self, x):
        return self.encoder(x)

    def classify(self, batch):
        B, T, H, W = batch.shape
        x = batch.unsqueeze(2).view(B * T, 1, H, W)
        x = self.forward(x)
        x = x.view(B, T, -1).mean(dim=1)
        return self.classifier(x)



class TemporalScanPredictor(nn.Module):
    def __init__(self, encoder, dim=256, image_size=224):
        super().__init__()
        self.encoder = encoder
        self.encoder.vit.mlp_head = nn.Identity()
        self.decoder = nn.Linear(dim, image_size * image_size)

    def forward(self, x):  # x: [B, 2, 1, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        feats = self.encoder.vit(x)
        feats = feats.view(B, T, -1).mean(dim=1)
        return self.decoder(feats).view(B, 1, H, W)