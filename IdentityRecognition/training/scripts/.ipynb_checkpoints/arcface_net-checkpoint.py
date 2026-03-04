# %% [code]
# ------------------------------------------------------------------
# This file builds ArcFace Net on (ResNet34)
# ------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.models as tv_models
# ---------------------------------------------------------------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    def __init__(self, num_classes, s = 64, margin = 0.2, embedding_dim = 512):
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
# ---------------------------------------------------------------------------------------------------------------------------------
class FaceRecognizerArcFace(nn.Module):
    def __init__(self, num_classes, backbone = None, embedding_dim = 512, margin = 0.2):
        # cfg
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.m = margin
        
        
        # backbone
        self.backbone = tv_models.resnet34(weights = tv_models.ResNet34_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_dim)

        # arc_face head
        self.arc_face = ArcFaceHead(num_classes = num_classes, embedding_dim = embedding_dim, margin = margin)

    def forward(self, x, labels = None, inference = False):
        embeddings = self.backbone(x)

        if inference:
            return F.normalize(embeddings, p = 2, dim = 1)
        
        logits = self.arc_face(embeddings, labels)
        return logits
