"""CNN and Vision Transformer image encoders for the VQA project.

Provides five encoder classes used by Models A–E:

- ``SimpleCNN``          (Model A) – 5-block scratch CNN → global vector
- ``SimpleCNNSpatial``   (Model C) – same backbone, keeps 7×7 spatial grid
- ``ResNetEncoder``      (Model B) – pretrained ResNet101 → global vector
- ``ResNetSpatialEncoder``(Model D) – pretrained ResNet101 → 7×7 spatial grid
- ``CLIPViTEncoder``     (Model E) – pretrained CLIP ViT-B/32 → 7×7 patch grid

All encoders output features of shape ``(B, hidden_size)`` (global) or
``(B, 49, hidden_size)`` (spatial), where 49 = 7×7.

Spatial Convention
------------------
The 7×7 = 49 spatial grid is produced by processing a 224×224 image through
5 MaxPool2d(stride=2) layers:
    224 → 112 → 56 → 28 → 14 → 7

For CLIP ViT-B/32 the 49 patches come from the Transformer's patch tokens
(patch_size=32, so 224/32 = 7 patches per side, 7×7 = 49).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPVisionModel


# ── Shared building block ────────────────────────────────────────────────────

def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build a standard convolutional block: Conv → BN → ReLU → MaxPool.

    ``padding=1`` on a 3×3 kernel preserves the spatial size before pooling,
    so five consecutive ``conv_block`` calls on a 224×224 input produce a
    7×7 feature map:  224 → 112 → 56 → 28 → 14 → 7.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.

    Returns:
        An ``nn.Sequential`` of (Conv2d, BatchNorm2d, ReLU, MaxPool2d).
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),  # halves spatial resolution
    )


# ── Model A encoder ──────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """Scratch CNN encoder that produces a single global image vector.

    Used by **Model A** (no attention, no pretrained backbone).

    Architecture::

        (B, 3,    224, 224)
          └─ conv_block(3→64)       → (B, 64,   112, 112)
          └─ conv_block(64→128)     → (B, 128,   56,  56)
          └─ conv_block(128→256)    → (B, 256,   28,  28)
          └─ conv_block(256→512)    → (B, 512,   14,  14)
          └─ conv_block(512→1024)   → (B, 1024,   7,   7)
          └─ AdaptiveAvgPool2d(1)   → (B, 1024,   1,   1)
          └─ flatten                → (B, 1024)
          └─ Linear(1024→output)    → (B, output_size)

    Args:
        output_size: Dimensionality of the output vector. Should match
            ``hidden_size`` of the question encoder and decoder.
    """

    def __init__(self, output_size: int = 1024) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3,    64),
            conv_block(64,   128),
            conv_block(128,  256),
            conv_block(256,  512),
            conv_block(512,  1024),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # squeeze (H, W) → (1, 1)
        self.fc   = nn.Linear(1024, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into global feature vectors.

        Args:
            x: ``FloatTensor (B, 3, 224, 224)`` — ImageNet-normalized images.

        Returns:
            ``FloatTensor (B, output_size)`` — one vector per image.
        """
        out = self.features(x)   # (B, 3, 224, 224) → (B, 1024, 7, 7)
        out = self.pool(out)     # (B, 1024, 7, 7)  → (B, 1024, 1, 1)
        out = out.flatten(1)     # (B, 1024, 1, 1)  → (B, 1024)
        out = self.fc(out)       # (B, 1024)         → (B, output_size)
        return out


# ── Model C encoder ──────────────────────────────────────────────────────────

class SimpleCNNSpatial(nn.Module):
    """Scratch CNN encoder that preserves spatial regions for attention.

    Used by **Model C** (Bahdanau attention, no pretrained backbone).
    Unlike ``SimpleCNN``, this encoder does NOT apply global average pooling,
    so the decoder can attend over the 49 spatial regions independently.

    Architecture::

        (B, 3,    224, 224)
          └─ conv_block(3→64)      → (B, 64,   112, 112)
          └─ conv_block(64→128)    → (B, 128,   56,  56)
          └─ conv_block(128→256)   → (B, 256,   28,  28)
          └─ conv_block(256→512)   → (B, 512,   14,  14)
          └─ conv_block(512→1024)  → (B, 1024,   7,   7)
          └─ Conv2d(1024→output, k=1) → (B, output_size, 7, 7)
          └─ flatten(dim=2)        → (B, output_size, 49)
          └─ permute(0,2,1)        → (B, 49, output_size)

    The 1×1 convolution is mathematically equivalent to a ``nn.Linear``
    applied independently to each of the 49 spatial positions.

    Args:
        output_size: Dimensionality of each region vector.
    """

    def __init__(self, output_size: int = 1024) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3,    64),
            conv_block(64,   128),
            conv_block(128,  256),
            conv_block(256,  512),
            conv_block(512,  1024),
        )
        # 1×1 conv projects each region from 1024 → output_size independently.
        self.proj = nn.Conv2d(1024, output_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into per-region feature vectors.

        Args:
            x: ``FloatTensor (B, 3, 224, 224)`` — ImageNet-normalized images.

        Returns:
            ``FloatTensor (B, 49, output_size)`` — 49 regional features.
        """
        out = self.features(x)   # (B, 3, 224, 224)      → (B, 1024, 7, 7)
        out = self.proj(out)     # (B, 1024, 7, 7)        → (B, output_size, 7, 7)
        out = out.flatten(2)     # (B, output_size, 7, 7) → (B, output_size, 49)
        out = out.permute(0, 2, 1)  # (B, output_size, 49) → (B, 49, output_size)
        return out


# ── Model B encoder ──────────────────────────────────────────────────────────

class ResNetEncoder(nn.Module):
    """Pretrained ResNet101 encoder that produces a global image vector.

    Used by **Model B** (no attention, pretrained backbone).

    ResNet101 children layout (0-indexed):
        0=conv1, 1=bn1, 2=relu, 3=maxpool,
        4=layer1, 5=layer2, 6=layer3, 7=layer4, 8=avgpool, 9=fc

    We remove the final ``fc`` layer (index 9), keeping ``avgpool`` to
    produce a ``(B, 2048, 1, 1)`` tensor, then project to ``output_size``.

    Args:
        output_size: Dimensionality of the output vector.
        freeze: If True, freeze all ResNet weights at init (Phase 1 training).
            Call ``unfreeze_top_layers()`` for Phase 2 fine-tuning.
    """

    def __init__(self, output_size: int = 1024, freeze: bool = True) -> None:
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # Remove only the final FC layer; keep avgpool for (B, 2048, 1, 1) output.
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(2048, output_size)

    def unfreeze_top_layers(self) -> None:
        """Selectively unfreeze layer3, layer4, and avgpool for Phase 2 fine-tuning.

        Keeps early layers (conv1, bn1, layer1, layer2) frozen to preserve
        low-level generic features. Only higher-level semantic layers adapt.
        ResNet children indices: 6=layer3, 7=layer4, 8=avgpool.
        """
        for param in self.resnet.parameters():
            param.requires_grad = False             # re-freeze everything first
        for child in list(self.resnet.children())[6:]:   # layer3, layer4, avgpool
            for param in child.parameters():
                param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def backbone_params(self) -> List[nn.Parameter]:
        """Return trainable backbone parameters for differential LR.

        Returns:
            List of ``nn.Parameter`` objects from layer3 + layer4 + avgpool.
        """
        return [
            p for child in list(self.resnet.children())[6:]
            for p in child.parameters()
            if p.requires_grad
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into global feature vectors.

        Args:
            x: ``FloatTensor (B, 3, 224, 224)`` — ImageNet-normalized images.

        Returns:
            ``FloatTensor (B, output_size)`` — one vector per image.
        """
        out = self.resnet(x)   # (B, 3, 224, 224) → (B, 2048, 1, 1)
        out = out.flatten(1)   # (B, 2048, 1, 1)  → (B, 2048)
        out = self.fc(out)     # (B, 2048)         → (B, output_size)
        return out


# ── Model D encoder ──────────────────────────────────────────────────────────

class ResNetSpatialEncoder(nn.Module):
    """Pretrained ResNet101 encoder that preserves spatial regions for attention.

    Used by **Model D** (Bahdanau attention, pretrained backbone).
    Unlike ``ResNetEncoder``, we remove **both** ``avgpool`` and ``fc``
    (last 2 children) to keep the full 7×7 spatial feature map.

    ResNet101 children layout:
        ..., 6=layer3, 7=layer4, [8=avgpool removed], [9=fc removed]
    Output of layer4: ``(B, 2048, 7, 7)``

    Args:
        output_size: Dimensionality of each region vector.
        freeze: If True, freeze all ResNet weights at init.
    """

    def __init__(self, output_size: int = 1024, freeze: bool = True) -> None:
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # Remove avgpool AND fc → output is spatial feature map (B, 2048, 7, 7).
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # 1×1 conv projects each region from 2048 → output_size independently.
        self.proj = nn.Conv2d(2048, output_size, kernel_size=1)

    def unfreeze_top_layers(self) -> None:
        """Selectively unfreeze layer3, layer4 for Phase 2 fine-tuning.

        ResNet children (without avgpool+fc): 6=layer3, 7=layer4.
        """
        for param in self.resnet.parameters():
            param.requires_grad = False
        for child in list(self.resnet.children())[6:]:   # layer3, layer4
            for param in child.parameters():
                param.requires_grad = True
        for param in self.proj.parameters():
            param.requires_grad = True

    def backbone_params(self) -> List[nn.Parameter]:
        """Return trainable backbone parameters for differential LR.

        Returns:
            List of ``nn.Parameter`` objects from layer3 + layer4.
        """
        return [
            p for child in list(self.resnet.children())[6:]
            for p in child.parameters()
            if p.requires_grad
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into per-region feature vectors.

        Args:
            x: ``FloatTensor (B, 3, 224, 224)`` — ImageNet-normalized images.

        Returns:
            ``FloatTensor (B, 49, output_size)`` — 49 regional features.
        """
        out = self.resnet(x)      # (B, 3, 224, 224)      → (B, 2048, 7, 7)
        out = self.proj(out)      # (B, 2048, 7, 7)        → (B, output_size, 7, 7)
        out = out.flatten(2)      # (B, output_size, 7, 7) → (B, output_size, 49)
        out = out.permute(0, 2, 1)  # (B, output_size, 49) → (B, 49, output_size)
        return out


# ── Model E encoder ──────────────────────────────────────────────────────────

class CLIPViTEncoder(nn.Module):
    """Pretrained CLIP ViT-B/32 encoder that produces spatial patch features.

    Used by **Model E** (FiLM fusion, CLIP backbone).

    CLIP ViT-B/32 processes 224×224 images with patch_size=32:
        224 / 32 = 7 patches per side → 7×7 = 49 patch tokens.

    The Transformer produces a sequence of length 50:
        index 0  : [CLS] token (global summary — discarded here)
        index 1–49 : patch tokens (spatial features — kept)

    The 768-d CLIP hidden size is projected to ``output_size`` via a
    learned linear layer.

    Args:
        output_size: Dimensionality of each patch feature vector.
        freeze: If True, freeze all CLIP weights at init (Phase 1 training).
            Call ``unfreeze_top_layers()`` for Phase 2 fine-tuning.
    """

    def __init__(self, output_size: int = 1024, freeze: bool = True) -> None:
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze:
            for param in self.clip.parameters():
                param.requires_grad = False
        # CLIP hidden size is 768; project each of the 49 patches to output_size.
        self.proj = nn.Linear(768, output_size)

    def unfreeze_top_layers(self) -> None:
        """Unfreeze the final Transformer encoder layer and the projection.

        Keeps all earlier layers frozen. CLIP's last encoder layer (index 11)
        captures the highest-level visual semantics most relevant to VQA.
        """
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.clip.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in self.proj.parameters():
            param.requires_grad = True

    def backbone_params(self) -> List[nn.Parameter]:
        """Return trainable backbone parameters for differential LR.

        Returns:
            List of ``nn.Parameter`` objects from the last CLIP encoder layer.
        """
        return [
            p for p in self.clip.vision_model.encoder.layers[-1].parameters()
            if p.requires_grad
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into per-patch feature vectors.

        Args:
            x: ``FloatTensor (B, 3, 224, 224)`` — ImageNet-normalized images.

        Returns:
            ``FloatTensor (B, 49, output_size)`` — 49 patch features.
        """
        # last_hidden_state: (B, 50, 768)  [CLS + 49 patch tokens]
        hidden_states = self.clip(x).last_hidden_state

        # Discard CLS token (index 0); keep the 49 spatial patch tokens.
        patch_features = hidden_states[:, 1:, :]  # (B, 50, 768) → (B, 49, 768)
        out = self.proj(patch_features)            # (B, 49, 768) → (B, 49, output_size)
        return out


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    B = 2
    x = torch.randn(B, 3, 224, 224)

    print("SimpleCNN:")
    print(f"  {SimpleCNN()(x).shape}")           # expect (2, 1024)

    print("SimpleCNNSpatial:")
    print(f"  {SimpleCNNSpatial()(x).shape}")    # expect (2, 49, 1024)

    print("ResNetEncoder:")
    print(f"  {ResNetEncoder(freeze=True)(x).shape}")         # expect (2, 1024)

    print("ResNetSpatialEncoder:")
    print(f"  {ResNetSpatialEncoder(freeze=True)(x).shape}")  # expect (2, 49, 1024)
