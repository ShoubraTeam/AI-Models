# %% [code]

# ---------------------------------------------------------------------------------------------------------------------------------------
# This file contanis important functions for model evaluation
# - Encoding Images
# - Calc Distances/Similarities  
# - Plot Distances/Similarities   
# - Evaluating Siamese/ArcFace
# - Calculting model thresholds & metrics
# - Evaluating on test examples
# ---------------------------------------------------------------------------------------------------------------------------------------



import torch
import numpy as np
import matplotlib.pyplot as plt
import  src.utils.config as config
from src.utils.functional import denormalize_img_tensor, tensor_to_ndarry, load_image_cv, crop_face
import torch.nn.functional as F
import random
import os
from PIL.Image import fromarray
import math


# -------------------------------------------------------------------------------------------------------------------------------------------
# Calculations
def encode_images(model, images, model_type = 'arcface'):
    """ 
    Use the trained model to encode the given images 

    Args:
        model: the trained model name
        images: a batch of image pairs
        model_type: str indicates whether the model is siamese or arcface
    """
    # cfg model
    model = model.to(config.DEVICE)
    model.eval()
    
    # extract data
    imgs1 = images[0].to(config.DEVICE)
    imgs2 = images[1].to(config.DEVICE)

    # pred
    with torch.no_grad():
        e1 = model(imgs1, inference = True)        
        e2 = model(imgs2, inference = True)        

    return e1, e2


def calc_cosine_similarity(encodings1, encodings2):
    """
    Calc the cosine similarity between two encodings for pais of images
    """
    encodings1 = encodings1.to(config.DEVICE)
    encodings2 = encodings2.to(config.DEVICE)
    
    s = F.cosine_similarity(encodings1, encodings2)
    return s    

# -------------------------------------------------------------------------------------------------------------------------------------------
# Testing on unseen data

def get_image_pairs(person_paths, n_images):
    images1 = []
    images2 = []
    labels = []
    
    for i in range(n_images):
        same_or_not = random.randint(0, 1)
        if same_or_not:
            random_person = random.choices(person_paths, k = 1)[0]
            img1_path = os.path.join(random_person, os.listdir(random_person)[0])
            img2_path = os.path.join(random_person, os.listdir(random_person)[1])
        
        else:
            random_persons = random.sample(person_paths, k = 2)
            person1 = random_persons[0]
            person2 = random_persons[1]
            img1_path = os.path.join(person1, os.listdir(person1)[0])
            img2_path = os.path.join(person2, os.listdir(person2)[0])

        img1 = load_image_cv(img1_path)
        img2 = load_image_cv(img2_path)
        images1.append(img1)
        images2.append(img2)
        labels.append(same_or_not)

    return images1, images2, labels   

def get_proximities(
    model,
    face_detector,
    images1: list,
    images2: list,
    labels: list,
    transform
):
    """
    Args:
        model: Torch trained model
        face_detector: a face detector model
        images: list of ndarrys
        labels: list of the labels
        transform: transformer

    Retuns:
        prox: distance or similarity between image embeddings
        faces1, faces2: cropped faces for visualization
        clean_labels: cleaned labels after cropping
    """

    # crop faces
    faces1 = [crop_face(face_detector, img) for img in images1]
    faces2 = [crop_face(face_detector, img) for img in images2]

    # remove Nones (if detector detected more than one image or no images at all)
    clean_faces1 = []
    clean_faces2 = []
    clean_labels = []
    for f1, f2, lab in zip(faces1, faces2, labels):
        if f1 is not None and f2 is not None:
            clean_faces1.append(f1)
            clean_faces2.append(f2)
            clean_labels.append(lab)

    # construct batches    
    clean_faces1 = [transform(fromarray(face)) for face in clean_faces1]
    clean_faces1 = torch.stack(clean_faces1, dim = 0)
    
    clean_faces2 = [transform(fromarray(face)) for face in clean_faces2]
    clean_faces2 = torch.stack(clean_faces2, dim = 0)

    # encode
    e1, e2 = encode_images(model = model, images = [clean_faces1, clean_faces2], model_type = "arcface") 

    # calc prox
    prox = calc_cosine_similarity(e1, e2)
    
    return prox, clean_faces1, clean_faces2, clean_labels



def plot_examples_with_similarities(images, similarities, threshhold, labels):
    """
    This function visualizes examples captioned with their distances and labels (if given)

    Args:
        images: a batch of image pairs [imgs1, imgs2]
                imgs1 (torch.size[batch_size, C, H, W])
                imgs2 (torch.size[batch_size, C, H, W])
        similarities: distances calculated among images (torch.size[batch_size])
        labels: labels indicating if there were the same persons or not (torch.size[batch_size])

    Returns:
        matches: number of correct classified examples
    """
    # Denormalize tensors
    imgs1 = torch.stack([denormalize_img_tensor(img) for img in images[0]])
    imgs2 = torch.stack([denormalize_img_tensor(img) for img in images[1]])

    # Turn tensors into ndarrays
    imgs1 = np.array([tensor_to_ndarry(img) for img in imgs1])
    imgs2 = np.array([tensor_to_ndarry(img) for img in imgs2])

    # construct plots
    same_or_not = {
        0 : "Different",
        1 : "Same"
    }

    n_samples = imgs1.shape[0]
    ncols = 4
    nrows = math.ceil(n_samples / ncols) 
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20, 20))
    axes_list = axes.flatten()

    matches = 0
    for img1, img2, sim, label, ax in zip(imgs1, imgs2, similarities, labels, axes_list):
        img_to_plot = np.concatenate((img1, img2), axis = 1)
        pred_label = 1 if sim.item() > threshhold else 0
        curr_label_int = int(label.item()) if torch.is_tensor(label) else int(label)
        is_correct = (pred_label == curr_label_int)
        color = 'green' if is_correct else 'red'
        if is_correct: 
            matches += 1
            
        title = f'-- {label} | similarity calculated: {sim.item():0.2f} -- '
            
        ax.imshow(img_to_plot)
        ax.set_title(title, color = color)
        ax.axis('off')


    for ax in axes.flat:
        if not ax.has_data(): fig.delaxes(ax)



    plt.tight_layout(h_pad = 2, w_pad = 0)
    plt.show()

    return matches

# -------------------------------------------------------------------------------------------------------------------------------------------
    
    
def calc_thresholds(
    model,
    loader,
    device = config.DEVICE
):
    """
    Calculates optimal thresholds and biometric metrics using vectorized cumulative sums.
    """

    import pandas as pd

    model = model.eval().to(device)
    proximities = []
    labels = []

    # encode
    with torch.no_grad():
        for (img1, img2), lab in tqdm(loader, desc = "Evaluating"):
            img1 = img1.to(device)
            img2 = img2.to(device)

            e1 = model(img1, inference = True)
            e2 = model(img2, inference = True)
            prox = F.cosine_similarity(e1, e2)

            proximities.extend(prox.detach().cpu().numpy())
            labels.extend(lab.detach().cpu().numpy())

    proximities = np.array(proximities)
    labels = np.array(labels)
    indices = np.argsort(proximities)[::-1]

    sorted_labels = labels[indices]
    sorted_proximities = proximities[indices]

    TPs = np.cumsum(sorted_labels)
    FPs = np.cumsum(1 - sorted_labels)

    total_pos = np.sum(labels)
    total_neg = len(labels) - total_pos

    FNs = total_pos - TPs
    TNs = total_neg - FPs

    FAR = FPs / (total_neg + 1e-7)  # FP_rate
    FRR = FNs / (total_pos + 1e-7)  # FN_rate
    ACC = (TPs + TNs) / len(labels)
    
    recall = TPs / (total_pos + 1e-7)
    precision = TPs / (TPs + FPs + 1e-7)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    eer_idx = np.nanargmin(np.abs(FAR - FRR))
    acc_idx = np.nanargmax(ACC)
    f1_idx  = np.nanargmax(F1)

    results = {
        'eer_threshold': (sorted_proximities[eer_idx]).item(),
        'eer_value': ((FAR[eer_idx] + FRR[eer_idx]) / 2).item(),
        'precision_at_eer': (precision[eer_idx]).item(),
        'recall_at_eer': (recall[eer_idx]).item(),
        
        'acc_threshold': (sorted_proximities[acc_idx]).item(),
        'best_accuracy': (ACC[acc_idx]).item(),
        
        'f1_threshold': (sorted_proximities[f1_idx]).item(),
        'best_f1': (F1[f1_idx]).item(),
        'precision_at_f1':(precision[f1_idx]).item(),
        'recall_at_f1': (recall[f1_idx]).item(),
    }

    return results



