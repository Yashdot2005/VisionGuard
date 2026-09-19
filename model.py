"""
VisionGuard - Neural Network Architectures
Day 3 Milestone: CNN architecture design, ResNet-18 pretrained transfer learning, adapt classifier head.

Architectures:
1. CustomCNN: 4-block Conv2D -> BatchNorm -> ReLU -> MaxPool with Dropout and Dense classification head.
2. VisionGuardResNet18: Pretrained ResNet-18 with frozen base layers, fine-tuned last two blocks,
   and custom dropout-regularized classifier head.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    """Custom Convolutional Neural Network for manufacturing defect detection.
    
    Structure:
    - Block 1: Conv2D(3 -> 32, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
    - Block 2: Conv2D(32 -> 64, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
    - Block 3: Conv2D(64 -> 128, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
    - Block 4: Conv2D(128 -> 256, 3x3) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2)
    - AdaptiveAvgPool2d((1, 1))
    - Dense Head: Linear(256 -> 128) -> ReLU -> Dropout(0.4) -> Linear(128 -> num_classes)
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


class VisionGuardResNet18(nn.Module):
    """ResNet-18 Transfer Learning model tailored for defect detection.
    
    Transfer Learning strategy:
    1. Base feature extractor initialized with ImageNet weights.
    2. Initial low-level layers (conv1, bn1, layer1, layer2) frozen.
    3. High-level feature blocks (layer3, layer4) kept trainable for fine-tuning defect patterns.
    4. Custom fully connected classification head with Dropout.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        fine_tune_last_blocks: bool = True,
        dropout_rate: float = 0.3
    ):
        super(VisionGuardResNet18, self).__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        if pretrained and fine_tune_last_blocks:
            # Freeze initial layers
            for param in base_resnet.parameters():
                param.requires_grad = False
            # Unfreeze layer3 and layer4 for fine-tuning
            for param in base_resnet.layer3.parameters():
                param.requires_grad = True
            for param in base_resnet.layer4.parameters():
                param.requires_grad = True
        elif pretrained and not fine_tune_last_blocks:
            # Feature extraction mode (freeze entire backbone)
            for param in base_resnet.parameters():
                param.requires_grad = False

        # Extract backbone (all layers up to avgpool)
        in_features = base_resnet.fc.in_features  # 512
        base_resnet.fc = nn.Identity()
        self.backbone = base_resnet

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def build_model(
    architecture: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    fine_tune_last_blocks: bool = True,
    dropout_rate: float = 0.3
) -> nn.Module:
    """Factory function to instantiate models based on architecture name."""
    arch_lower = architecture.lower()
    if arch_lower in ["resnet18", "resnet"]:
        model = VisionGuardResNet18(
            num_classes=num_classes,
            pretrained=pretrained,
            fine_tune_last_blocks=fine_tune_last_blocks,
            dropout_rate=dropout_rate
        )
    elif arch_lower in ["custom_cnn", "custom", "cnn"]:
        model = CustomCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose 'resnet18' or 'custom_cnn'.")

    return model


def get_model_summary(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    print("Testing Model Architectures:")
    x = torch.randn(2, 3, 224, 224)

    # Test Custom CNN
    cnn = build_model(architecture="custom_cnn", num_classes=2)
    cnn_out = cnn(x)
    total_c, train_c = get_model_summary(cnn)
    print(f"\n[Custom CNN] Output shape: {cnn_out.shape} | Total Params: {total_c:,} | Trainable: {train_c:,}")

    # Test ResNet-18 Transfer Learning
    resnet = build_model(architecture="resnet18", num_classes=2, pretrained=True)
    resnet_out = resnet(x)
    total_r, train_r = get_model_summary(resnet)
    print(f"[ResNet-18]  Output shape: {resnet_out.shape} | Total Params: {total_r:,} | Trainable: {train_r:,}")
    print("\nModel definitions verified successfully!")
