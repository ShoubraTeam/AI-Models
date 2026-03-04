# %% [code]

# ---------------------------------------------------------------------------------------------------------------------------------------
# This file contanis important functions for model evaluation
# - Encoding Images
# - Calc Euclidean Distances
# - Visualize Examples
# - Get best distance threshold for Siamese network
# ---------------------------------------------------------------------------------------------------------------------------------------



import torch
import numpy as np
import matplotlib.pyplot as plt
import scripts.config as config
# import face_recognition_config as config
from scripts.utils import denormalize_img_tensor, tensor_to_ndarry, load_image_cv, crop_face
# from face_recognition_utils import denormalize_img_tensor, tensor_to_ndarry
import torch.nn.functional as F
import random
import os
from PIL.Image import fromarray
from sklearn.metrics import roc_curve, auc
import math

# ---------------------------------------------------------------------------------------------------------------------------------------
def encode_images(model, images, model_type = 'siamese'):
    """ 
    Use the trained model to encode the given images 

    Args:
        model: the trained model (siamese, arcface)
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
        if model_type.lower() == 'siamese':
            e1, e2 = model(imgs1, imgs2)
            e1, e2 = F.normalize(e1), F.normalize(e2)
            
        elif model_type.lower() == 'arcface':
            e1 = model(imgs1, inference = True)        
            e2 = model(imgs2, inference = True)        

        else:
            raise ValueError(f"Not model called {model_type}")
    
    return e1, e2
# ---------------------------------------------------------------------------------------------------------------------------------------
def calc_euclidean_distance(encodings1, encodings2):
    """
    Calc the distances between two output encodings for pair of images
    """
    encodings1 = encodings1.to(config.DEVICE)
    encodings2 = encodings2.to(config.DEVICE)
    
    d = F.pairwise_distance(encodings1, encodings2, keepdim = False)
    return d
# ---------------------------------------------------------------------------------------------------------------------------------------
def calc_cosine_similarity(encodings1, encodings2):
    """
    Calc the cosine similarity between two output encodings for pais of images
    """
    encodings1 = encodings1.to(config.DEVICE)
    encodings2 = encodings2.to(config.DEVICE)
    
    s = F.cosine_similarity(encodings1, encodings2)
    return s
# ---------------------------------------------------------------------------------------------------------------------------------------
def plot_examples_with_distances(images, distances, threshhold, labels):
    """
    This function visualizes examples captioned with their distances and labels (if given)

    Args:
        images: a batch of image pairs [imgs1, imgs2]
                imgs1 (torch.size[batch_size, C, H, W])
                imgs2 (torch.size[batch_size, C, H, W])
        distances: distances calculated among images (torch.size[batch_size])
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
    for img1, img2, distance, label, ax in zip(imgs1, imgs2, distances, labels, axes_list):
        img_to_plot = np.concatenate((img1, img2), axis = 1)

        if isinstance(label, torch.Tensor): true_label = same_or_not[label.int().item()]
        else: true_label = same_or_not[label]
        
        pred_label = 0 if distance.item() > threshhold else 1
        color = 'green' if pred_label == label else 'red'
        title = f'-- {label} | distance calculated: {distance.item():0.2f} -- '
            
        ax.imshow(img_to_plot)
        ax.set_title(title, color = color)
        ax.axis('off')

        if pred_label == label: matches += 1

    for ax in axes.flat:
        if not ax.has_data(): fig.delaxes(ax)

    plt.tight_layout(h_pad = 2, w_pad = 0)
    plt.show()

    return matches
# ---------------------------------------------------------------------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------------------------------------------------------------------
def get_subset_distances(model, loader, max_batches = 40, normalize = False):
    """
    Calc the distance for (max_batches) of encodings that the model outputs

    Args:
        model: trained Torch model
        loader: Torch dataloader
        max_batches: number of max_batches to evaluate the model on

    Returs:
        distances: distances calculated for all modeled examples np.array(max_batches * batch_size)
        labels: labels for all modeled examples np.array(max_batches * batch_size)
    """

    distances = []
    labels = []
    batches_modeled = 0

    for images, lab in loader:
        if batches_modeled >= max_batches:
            break

        encodings1, encodings2 = encode_images(model, images)
        if normalize:
            encodings1 = F.normalize(encodings1, p = 2, dim = 1)
            encodings2 = F.normalize(encodings2, p = 2, dim = 1)
            
        distance = calc_euclidean_distance(encodings1, encodings2)
        

        distances.extend(distance.detach().cpu().numpy())
        labels.extend(lab.numpy())
        batches_modeled += 1

    return np.array(distances), np.array(labels)
# ---------------------------------------------------------------------------------------------------------------------------------------
def get_subset_similarities(model, loader, max_batches = 40, device = config.DEVICE):
    """
    Calc the sim for (max_batches) of encodings that the model outputs

    Args:
        model: trained Torch model
        loader: Torch dataloader
        max_batches: number of max_batches to evaluate the model on

    Returs:
        similarities: similarities calculated for all modeled examples np.array(max_batches * batch_size)
        labels: labels for all modeled examples np.array(max_batches * batch_size)
    """

    similarities = []
    labels = []
    batches_modeled = 0

    for images, lab in loader:
        if batches_modeled >= max_batches:
            break
            
        imgs1 = images[0].to(device)
        imgs2 = images[1].to(device)
        
        encodings1 = model(imgs1)
        encodings2 = model(imgs2)
            
        sim = F.cosine_similarity(encodings1, encodings2)
        

        similarities.extend(sim.detach().cpu().numpy())
        labels.extend(lab.numpy())
        batches_modeled += 1

    return np.array(similarities), np.array(labels)
# --------------------------------------------------------------------------------------------------------------------------------------
def get_best_threshold(distances, labels, n_steps = 200):
    """
    Finding the best threshold that maximizes accuracy on validation set for Siamese
    """

    thresholds = np.linspace(distances.min(), distances.max(), n_steps)

    best_th = 0
    best_acc = 0

    for t in thresholds:
        preds = (distances < t).astype(int)
        accuracy = np.mean(preds == labels)

        if accuracy > best_acc:
            best_acc = accuracy
            best_th = t
    return best_th, best_acc
# ---------------------------------------------------------------------------------------------------------------------------------------
# ## Evaluating ArcFace ##
def predict_labels_on_train(model, images, true_labels, device = config.DEVICE):
    """
    Args:
        model: trained ArcFace
        images: batch of images [b, c, h, w]
        true_labels: batch of labels [b]
    Returns:
        predictions: [b]
    """
    model.eval().to(device)
    with torch.no_grad():
        images = images.to(device)
        true_labels = true_labels.to(device)
        logits = model(images, true_labels)
        predicted = torch.argmax(logits, dim = 1)
    return predicted

# predict on val -> using embeddings
def predict_labels_on_val(model, images, device = config.DEVICE):
    """
    Args:
        model: trained ArcFace
        images: batch of images [b, c, h, w]

    Returns:
        predicted person labels [b]
    """
    model.eval().to(device)
    with torch.no_grad():
        images = images.to(device)

        # encode images 
        embeddings = model(images, inference = True) # [b, emb]

        # get model weights (each weight represents a person's center)
        weights = F.normalize(model.arc_face.weight, p = 2, dim = 1) # [n_persons, emb]

        # preds
        logits = torch.mm(embeddings, weights.t())   # [b, n_persons]
        predicted = torch.argmax(logits, dim = 1)
        
        return predicted


# ---------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------  Evaluating Models --------------------------------------------
def calc_thresholds(
    model,
    loader,
    proximity_type: str,
    normalize_on_distance = False,
    device = config.DEVICE
):
    """
    Calculates optimal thresholds and biometric metrics using vectorized cumulative sums.
    """

    import pandas as pd

    model = model.eval().to(device)
    proximities = []
    labels = []

    if proximity_type.lower() not in ['distance', 'cosine_similarity']:
        raise ValueError("The given proximity type is not allowed")

    # encode
    with torch.no_grad():
        for images, labs in loader:
            imgs1 = images[0].to(device)
            imgs2 = images[1].to(device)

            if proximity_type == 'distance':
                e1, e2 = model(imgs1, imgs2)
                if normalize_on_distance:
                    e1 = F.normalize(e1, p=2, dim=1)
                    e2 = F.normalize(e2, p=2, dim=1)
                prox = torch.norm(e1 - e2, dim = 1)
                
            else:
                e1 = model(imgs1, inference = True)
                e2 = model(imgs2, inference = True)
                prox = F.cosine_similarity(e1, e2)

            proximities.extend(prox.detach().cpu().numpy())
            labels.extend(labs.detach().cpu().numpy())

    proximities = np.array(proximities)
    labels = np.array(labels)

    if proximity_type == 'distance':
        indices = np.argsort(proximities)
    else:
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



def plot_roc_curve(prox: list, labels: list, subsets: list, model_name, is_similarity = False):
    """
    This function plots the ROC curve over all possible prox thresholds to measure the model generalization capability.

    Args:
        prox: calculated prox among pair of images, list of (ndarry (n_pairs,)).
        labels: true labels of the given pair of images, list (ndarry (n_pairs,))
        subsets: any combination of train, val, and test sets, list of strings.
    """

    assert len(prox) == len(subsets) == len(labels), "Input lists must be same length"
    
    plt.figure(figsize = (10, 7))
    colors = ['blue', 'green', 'red']


    # eer diagonal
    plt.plot([0, 1], [1, 0], color = 'gray', linestyle = '--', alpha = 0.5, label = 'EER Line (FAR=FRR)')


    EERs = []
    AUCs = []
    for idx, (s, p, l, c) in enumerate(zip(subsets, prox, labels, colors)):
        # calc
        score = p if is_similarity else -p
        FPR, TPR, _ = roc_curve(y_true = l, y_score = score) 
        roc_auc = auc(x = FPR, y = TPR)
        
        # roc curve
        plt.plot(FPR, TPR, color = c, lw = 2, label = f'{s} (AUC = {roc_auc:.4f})')

        # EER point
        FAR = FPR
        FRR = 1 - TPR
        EER_idx = np.nanargmin(np.absolute(FAR - FRR))
        x, y = FPR[EER_idx], TPR[EER_idx]
        
        # Use color-coded markers instead of black to distinguish subsets
        plt.scatter(x = x, y = y, color = c, s = 60, zorder = 5, edgecolors = 'white')
        
        # Use a staggered offset for text based on index to prevent stacking
        # Alternates text above or below the point
        y_offset = 0.02 if idx % 2 == 0 else -0.08
        x_offset = 0.10
        
        plt.annotate(
            f'{s} EER: {x:.2%}', 
            xy = (x, y), 
            xytext = (x + x_offset, y + y_offset),
            fontsize = 10,
            fontweight = 'bold',
            color = c,
            arrowprops = dict(arrowstyle = "->", color = c, connectionstyle = "arc3, rad=.2")
        )
        
        EERs.append(x)
        AUCs.append(roc_auc)
        
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR | FAR)')
    plt.ylabel('True Positive Rate (TPR | 1 - FRR)')
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC)')
    plt.legend(loc = "lower right", frameon = True, shadow = True)
    plt.grid(alpha = 0.3, linestyle = ':')
    plt.tight_layout()
    plt.show()


    return EERs, AUCs


def plot_proximity_historgram(prox: list, labels: list, subsets: list, threshold, model_name, prox_label = 'distance'):
    """
    This function viusalizes the histogram of calculated prox for each subset in subsets (train, val, test)

    Args:
        prox: calculated proximities among pair of images, list of (ndarry (n_pairs,)).
        labels: true labels of the given pair of images, list (ndarry (n_pairs,))
        subsets: any combination of train, val, and test sets, list of strings.
        threshold: the float distance that distinguishes same persons from different persons
    """

    assert len(prox) == len(subsets), "Lenghth of prox arrays should match the subsets"
    assert len(labels) == len(subsets), "Lenghth of prox arrays should match the subsets"
    
    _, axes = plt.subplots(nrows = 1, ncols = len(subsets), figsize = (len(subsets) * 6, 5))
    for ax, p, l, s in zip(axes, prox, labels, subsets):
        ax.hist(p[l == 1], bins = 30, alpha = 0.5, label = "Same", color = 'green')
        ax.hist(p[l == 0], bins = 30, alpha = 0.5, label = "Different", color = 'red')
        ax.axvline(x = threshold, color = 'blue', alpha = 1, label = f'Threshold: {threshold:.2f}')
        ax.set_title(f'{model_name} Distances Histogram on {s.capitalize()}')
        ax.set_xlabel(prox_label.capitalize())
        ax.legend()
        

    plt.tight_layout()
    plt.show()


def get_subset_similarities(model, loader, max_batches = 40, device = config.DEVICE):
    """
    Calc the sim for (max_batches) of encodings that the model outputs

    Args:
        model: trained Torch model
        loader: Torch dataloader
        max_batches: number of max_batches to evaluate the model on

    Returs:
        similarities: similarities calculated for all modeled examples np.array(max_batches * batch_size)
        labels: labels for all modeled examples np.array(max_batches * batch_size)
    """

    similarities = []
    labels = []
    batches_modeled = 0
    model = model.to(config.DEVICE)

    with torch.no_grad():
        for images, lab in loader:
            if batches_modeled >= max_batches:
                break
                
            imgs1 = images[0].to(device)
            imgs2 = images[1].to(device)
            
            encodings1 = model(imgs1)
            encodings2 = model(imgs2)
                
            sim = F.cosine_similarity(encodings1, encodings2)
            
    
            similarities.extend(sim.detach().cpu().numpy())
            labels.extend(lab.numpy())
            batches_modeled += 1

            del imgs1, imgs2, encodings1, encodings2, sim

    return np.array(similarities), np.array(labels)
# ---------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Evaluating on Test Set ------------------------------------------------------
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
    model_type = 'siamese',
):
    """
    Args:
        model: Torch trained model
        face_detector: a face detector model
        images: list of ndarrys
        labels: list of the labels
        model_type: str indices whether a siamese or arcface is the model
        normalize_on_distance: whether to normalize on distance or not

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
    clean_faces1 = [config.val_transform(fromarray(face)) for face in clean_faces1]
    clean_faces1 = torch.stack(clean_faces1, dim = 0)
    
    clean_faces2 = [config.val_transform(fromarray(face)) for face in clean_faces2]
    clean_faces2 = torch.stack(clean_faces2, dim = 0)

    # encode
    e1, e2 = encode_images(model = model, images = [clean_faces1, clean_faces2], model_type = model_type) 

    # calc prox
    if model_type == 'siamese':
        prox = calc_euclidean_distance(e1, e2)
        
    elif model_type == 'arcface':
        prox = calc_cosine_similarity(e1, e2)
    
    return prox, clean_faces1, clean_faces2, clean_labels

def get_proximities_tta(
    model,
    face_detector,
    images1: list,
    images2: list,
    labels: list,
    transforms,
    model_type = 'siamese',
):
    """
    Args:
        model: Torch trained model
        face_detector: a face detector model
        images: list of ndarrys
        labels: list of the labels
        model_type: str indices whether a siamese or arcface is the model
        transforms: transforms for use in augmentation

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
    proximities = []
    for idx, transform in enumerate(transforms):
        images_1 = torch.stack(
            tensors = [transform(fromarray(face)) for face in clean_faces1],
            dim = 0
        )

        images_2 = torch.stack(
            tensors = [transform(fromarray(face)) for face in clean_faces2],
            dim = 0
        )

        
        # encode
        e1, e2 = encode_images(model = model, images = [images_1, images_2], model_type = model_type) 

        if model_type == 'arcface':
            prox = calc_cosine_similarity(e1, e2)
            proximities.append(prox)
            
        else:
            prox = calc_euclidean_distance(e1, e2)
            proximities.append(prox)


        # to get images
        if idx == 0:
            f1 = images_1
            f2 = images_2
                
    prox = sum(proximities) / len(proximities)
    
    return prox, f1, f2, clean_labels