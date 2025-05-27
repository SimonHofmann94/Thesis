import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    VGG11_Weights,
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    DenseNet121_Weights
)

class ClassifierModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'vgg11',
        pretrained: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()

        # VGG11 Backbone
        if backbone_name == 'vgg11':
            weights = VGG11_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.vgg11(weights=weights)
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            in_features = backbone.classifier[-1].in_features
            classifier_layers = list(backbone.classifier.children())[:-1]
            classifier_layers.append(nn.Linear(in_features, num_classes))
            self.classifier = nn.Sequential(*classifier_layers)

        # ResNet18 / ResNet50 Backbone
        elif backbone_name in ['resnet18', 'resnet50']:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = getattr(models, backbone_name)(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.features = backbone
            self.avgpool = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)

        # EfficientNet-B0 Backbone
        elif backbone_name.startswith('efficientnet'):
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = getattr(models, backbone_name)(weights=weights)
            in_features = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            self.features = backbone
            self.avgpool = nn.Identity()
            self.classifier = nn.Linear(in_features, num_classes)

        # DenseNet121 Backbone
        elif backbone_name == 'densenet121':
            weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.densenet121(weights=weights)
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.features = backbone.features
            # DenseNet liefert bereits 1x1 Feature-Map
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unbekannter Backbone: {backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Pooling (falls nötig)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
