import torch.nn as nn
from vit_pytorch import ViT
import torch


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
            channels=1,
            dropout=0.2,
            emb_dropout=0.2
        )

        self.dim = dim
        self.encoder = self.vit
        self.encoder.mlp_head = nn.Identity()
        self.classifier = nn.Linear(dim, 6)

        print("Patch embedding Conv2d:")
        print(self.vit.to_patch_embedding[0])  # Should be Conv2d(1, dim, ...)

        print("\nLayerNorm at the end of encoder:")
        print(self.vit.to_patch_embedding)  # Should be LayerNorm(dim)

        print("\nMLP head:")
        print(self.vit.mlp_head)

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
        self.dim = dim
        self.decoder = nn.Linear(dim, image_size * image_size)

    def forward(self, x):  # x: [B, 2, 1, H, W]
        B, T, _, C, H, W = x.shape

        # Rearrange to [B, T, C, 1, H, W] → flatten to [B * T * C, 1, H, W]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, 1, C, T, H, W]
        x = x.view(B * T * C, 1, H, W)                # [B*T*C, 1, H, W]

        # Encode each slice individually
        feats = self.encoder(x)                   # [B*T*C, N_patches, dim]

        # Reshape back to [B, T*C, dim], then mean across T*C → [B, dim]
        print("feats after encoder:", feats.shape)
        print("feats total elements:", feats.numel())
        print("Expected total:", B * T * C * self.dim)
        feats = feats.view(B, T * C, self.dim).mean(dim=1)  # [B, dim]

        # Decode to [B, 1, H, W]
        out = self.decoder(feats).view(B, 1, H, W)
        return out
