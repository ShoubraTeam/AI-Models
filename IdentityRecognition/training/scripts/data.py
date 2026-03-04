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



class SiameseDataset(Dataset):
    def __init__(self, data_path: str, transform = None, epoch_size = None):
        """
        Initializes a siamese network dataset
        Args:
            data_path: The main path to the faces images
            transform: TorchVision transformations
            epoch_size: number of pairs to generate per epoch (to control training)
        """
        self.data_path = data_path
        self.transform = transform
        self.epoch_size = epoch_size
        self.person_dict = self.__create_person_dict()
        self.persons = list(self.person_dict.keys())

        # For errors
        self.n_errors = 0
        
    # ----------------------------------------------------------------------------------
    def __create_person_dict(self):
        """
        Creating a dictionary that maps each person to a list of their image paths
        """
        person_dict = {}
        for person in os.listdir(self.data_path):
            # person dir
            person_path = os.path.join(self.data_path, person)

            if os.path.isdir(person_path):

                if len(os.listdir(person_path)) >= 2:
                    image_paths = [
                        os.path.join(person_path, img_path) for img_path in os.listdir(person_path)
                    ]
                    person_dict[person] = image_paths
        return person_dict
    # ----------------------------------------------------------------------------------
    def __len__(self):
        return self.epoch_size
    # ----------------------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        Fetches the requested data sample and returns it
        """
        # Label
        same_or_not = random.randint(0, 1)

        try:
            # Images
            if same_or_not:
                img1, img2 = self._generate_positive_sample(idx)
            else:
                img1, img2 = self._generate_negative_sample(idx)
            
        except Exception as e:
            print(f"Erros loading images: {e}")
            self.n_errors += 1
            return self.__getitem__(random.randint(0, self.epoch_size - 1))
            
        try:
            # Transformation
            if not self.transform:
                raise ValueError("ToTensor transform must be used")
                
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        except Exception as e:
            print(f"Erros transforming images: {e}")
            self.n_errors += 1
            return self.__getitem__(random.randint(0, self.epoch_size - 1))
        
        x = [img1, img2]
        y = torch.tensor(same_or_not, dtype = torch.float32)
        return x, y
        
    # ----------------------------------------------------------------------------------
    def _generate_positive_sample(self, idx):
        """
        Loads two images for a single preson
        """
        # choose a person
        random_person = random.choices(
            population = self.persons,
            k = 1 # retrieves only one person
        )[0]

        
        # get images
        random_img_paths = random.sample( # sample without replacement
            population = self.person_dict[random_person],
            k = 2
        )
        

        img1 = self.load_image_pil(random_img_paths[0])
        img2 = self.load_image_pil(random_img_paths[1])
        return img1, img2
    # ----------------------------------------------------------------------------------
    def _generate_negative_sample(self, idx):
        """
        Loads two images for two different presons
        """
        # choose persons
        random_persons = random.sample(
            population = self.persons,
            k = 2 # retrieves two persons
        )
    
        # get images
        person1_img_path = random.choices(
            population = self.person_dict[random_persons[0]],
            k = 1
        )[0]
    
        person2_img_path = random.choices(
            population = self.person_dict[random_persons[1]],
            k = 1
        )[0]
           
        img1 = self.load_image_pil(person1_img_path)
        img2 = self.load_image_pil(person2_img_path)
        
    
        return img1, img2
    # ----------------------------------------------------------------------------------
    def load_image_pil(self, path):
        """ Loads an image using PIL format """
        img = PIL.Image.open(path).convert("RGB")
        return img
    # ----------------------------------------------------------------------------------
    def log_errors(self):
        pass
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# ArcFace datasets [Used in Training]
class ArcFaceDataset(torch.utils.data.Dataset):
    def __init__(self, data, indices, labels, transform = None):
        """
        Args:
            data: the original full dataset
            indices: the subset indices
            labels: subset mapped new labels after removing the persons that have <2 images
        """
        
        self.data = data
        self.indices = indices
        self.labels = labels
        self.transform = transform
    

    def __getitem__(self, idx):
        org = self.indices[idx]
        x, _ = self.data[org]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.indices)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Pairs data [used in evaluation]
class ArcFaceDataset2(torch.utils.data.Dataset):
    def __init__(self, data, indices, labels, transform = None):
        """
        Args:
            data: the original full dataset
            indices: the subset indices
            labels: subset mapped new labels after removing the persons that have <2 images
        """
        
        self.data = data
        self.indices = indices
        self.labels = labels
        self.transform = transform
    
    def get_label(self, idx):
        return self.labels[idx]

    def __getitem__(self, idx):
        org = self.indices[idx]
        x, _ = self.data[org]
        y = self.labels[idx] 
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

class OpenSetArcfaceDataset(torch.utils.data.Dataset):
    " Constructs Same-Different Pairs """
    
    def __init__(self, arcface_dataset, n_pairs):
        self.arcface_dataset = arcface_dataset
        self.n_pairs = n_pairs
        self.person_to_indices = self.generate_person_dict()
        self.all_labels = list(self.person_to_indices.keys())
        self.labels_with_pairs = [p for p in self.all_labels if len(self.person_to_indices[p]) >= 2] # filter for (same) person to ensure that there are more than one image
        self.pairs, self.targets = self.generate_pairs()

    def generate_person_dict(self):
        person_to_indices = {}
        for idx in range(len(self.arcface_dataset)):
            label = self.arcface_dataset.get_label(idx) 
            if label not in person_to_indices:
                person_to_indices[label] = []
            person_to_indices[label].append(idx)
        return person_to_indices
        

    def generate_pairs(self):
        pairs = []
        targets = []
        
        # same
        for _ in range(self.n_pairs // 2):
            person = random.choice(self.labels_with_pairs)
            idx1, idx2 = random.sample(self.person_to_indices[person], k = 2)
            pairs.append((idx1, idx2))
            targets.append(1)

        # diff
        for _ in range(self.n_pairs // 2):
            person1, person2 = random.sample(self.all_labels, k = 2)
            idx1 = random.choice(self.person_to_indices[person1])
            idx2 = random.choice(self.person_to_indices[person2])
            pairs.append((idx1, idx2))
            targets.append(0)

        return pairs, targets

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        img1, _ = self.arcface_dataset[idx1]
        img2, _ = self.arcface_dataset[idx2]
        return (img1, img2), self.targets[idx]

    def __len__(self):
        return self.n_pairs