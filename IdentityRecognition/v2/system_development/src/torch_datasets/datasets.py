# %% [code]
# %% [code]
# %% [code]
# %% [code]
# ------------------------------------------------------------------
# This file Contains the datasets used
# ------------------------------------------------------------------

import os
import random
import PIL
import torch
from torch.utils.data import Dataset

class ArcFaceDatasetV2(Dataset):
    """
    Class for builing a torch-compatible dataset for training an ArcFace model.

    Args:
        dataset    : the subset of samples
        transformer: TorchVision transform pipeline
    """
    def __init__(self, dataset, transformer = None):
        self.dataset = dataset
        self.transform = transformer

    def __len__(self):
        return len(self.dataset)
    # ---------------------------------------------------------------------------------------------------
    def prepare_img(self, idx):
        """Prepare images"""
        # get img
        img = np.array(self.dataset[idx][0])

        # transform
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)
        return img
    # ---------------------------------------------------------------------------------------------------
    def __getitem__(self, idx):
        face  = self.prepare_img(idx)
        label = torch.tensor(self.dataset[idx][1], dtype = torch.long)
        return face, label
    # ---------------------------------------------------------------------------------------------------    

class ArcFaceV2PairsDataset(Dataset):
    """Used for constructing pairs data from a given subset"""
    def __init__(self, dataset, transform = None, pairs_per_epoch = 20000):
        self.dataset = dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.transform = transform

        # group indices by class
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            label = int(label.item())
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        if random.random() < 0.5:
            # same person
            cls = random.choice(self.classes)
            indices = self.class_to_indices[cls]

            if len(indices) < 2:
                return self.__getitem__(idx)

            sample1, sample2 = random.sample(indices, 2)

            img1, _ = self.dataset[sample1]
            img2, _ = self.dataset[sample2]

            label = 1

        else:
            # different persons
            cls1, cls2 = random.sample(self.classes, 2)

            sample1 = random.choice(self.class_to_indices[cls1])
            sample2 = random.choice(self.class_to_indices[cls2])

            img1, _ = self.dataset[sample1]
            img2, _ = self.dataset[sample2]

            label = 0

        
        return (img1, img2), torch.tensor(label, dtype = torch.float32)