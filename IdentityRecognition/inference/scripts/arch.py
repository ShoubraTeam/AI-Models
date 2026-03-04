# --------------------------------------------------------------------------------
# This file contains the implementations for Siamese & ArcFace
# --------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride=stride, padding=1, bias = False)

# -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- IResNet Backbone ---------------------------------------
class IBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

class IResNet(nn.Module):
    def __init__(self, block, layers, num_features=512):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=0)
        self.fc = nn.Linear(512 * 7 * 7, num_features) # Matches 112x112 input
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x




# -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- IResNet Siamese ---------------------------------------
class IResNetSiameseNetwork(nn.Module):
    def __init__(self, encoding_dimensions = 128, dropout_rate = 0.4, backbone_weights = False):
        super().__init__()
        
        self.backbone = IResNet(IBasicBlock, [3, 4, 14, 3])
        if backbone_weights:
            self.backbone.load_state_dict(
                torch.load("/kaggle/input/iresnet-34-50-100-weights/pytorch/default/1/iresnet50-7f187506.pth", map_location = 'cpu')
            )

        backbone_features = self.backbone.features.num_features
        self.fc = nn.Sequential(
            nn.Linear(backbone_features, encoding_dimensions)
        )

    def forward(self, img1, img2):
        return self.encode_img(img1), self.encode_img(img2)

    def encode_img(self, img):
        x = self.backbone(img)
        x = self.fc(x) 
        x = F.normalize(x, p = 2, dim = 1)
        return x


# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- IResNet ArcFace ---------------------------------------------------

class ArcFaceHead(nn.Module):
    def __init__(self, num_classes, s = 64, margin = 0.5, embedding_dim = 512):
        """
        Declaring the ArcFace model

        Args:
            num_classes: num of output classes
            s: the scaling factor
            margin: the cosine margin factor 
            embedding_dim: embedding_dimensions
            dropout_rate: to prevent overfitting
        """
        # cfg
        super().__init__()

        self.num_classes = num_classes
        self.s = s
        self.m = margin
        self.embedding_dim = embedding_dim

        # weight
        self.weight = nn.Parameter(data = torch.FloatTensor(num_classes, embedding_dim))  # [n_classes, emb] >> n_classes == n
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels = None):
        """
        Args:
            embeddings: embeddings extracted from the backbone [b, emb]
            labels: ground truth labels, needed for margin loss [b]
        """
        # normalize 
        X = F.normalize(embeddings, p = 2, dim = 1)      # [b, emb]
        W = F.normalize(self.weight, p = 2, dim = 1)     # [n, emb]

        # cosine similarity (dot product)
        cosine = torch.mm(X, W.t())  # [b, n]

        if labels is None:
            return cosine * self.s
        
        eps = 1e-7
        cosine = cosine.clamp(-1.0 + eps, 1.0 - eps)

        # angle
        theta = torch.acos(cosine)

        # add margin
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes = self.num_classes).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot

        # scale
        logits *= self.s

        return logits



class FaceRecognizerArcFace(nn.Module):
    def __init__(self, num_classes, embedding_dim = 512, margin = 0.5, backbone_weights = False):
        # cfg
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.m = margin
        
        
        # backbone
        self.backbone = IResNet(IBasicBlock, [3, 4, 14, 3])
        if backbone_weights:
            self.backbone.load_state_dict(
                torch.load("/kaggle/input/iresnet-34-50-100-weights/pytorch/default/1/iresnet50-7f187506.pth", map_location = 'cpu')
            )

        # arc_face head
        self.arc_face = ArcFaceHead(num_classes = num_classes, embedding_dim = embedding_dim, margin = margin)

    def forward(self, x, labels = None, inference = False):
        embeddings = self.backbone(x)

        if inference:
            return F.normalize(embeddings, p = 2, dim = 1)
        
        logits = self.arc_face(embeddings, labels)
        return logits

