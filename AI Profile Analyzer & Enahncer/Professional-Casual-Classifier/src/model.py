import torch.nn as nn
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class TransferVGG16(nn.Module):
    def __init__(self):
        super(TransferVGG16, self).__init__()
        self.base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Freeze Layers
        for param in self.base_model.features.parameters():
            param.requires_grad = False
            
        # Replace Last Layer
        num_ftrs = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.base_model(x)


def get_model(model_name):
    if model_name == "baseline_cnn":
        return SimpleCNN()
    elif model_name == "vgg16_transfer":
        return TransferVGG16()
    else:
        raise ValueError(f"Unknown Model: {model_name}")