# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# %% [code]
# ------------------------------------------------------------------
# This file implements necessary functions for training:
# - get data loaders
# - train epoch
# - val epoch
# - save & load model checkpoints & results
# ------------------------------------------------------------------
import sys



import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scripts.config as config
from scripts.evaluation import get_best_threshold
from scripts.utils import json_safe, save_obj


# ------------------------------------------------------------------------------------------------------------------------
def get_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size = 64,
    num_workers = 4, 
):
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )
    
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )

    return train_dataloader, val_dataloader, test_dataloader
# ------------------------------------------------------------------------------------------------------------------------
def load_checkpoint(path, model, optimizer = None, scheduler = None, device = 'cpu'):
    """
    Load a trained model

    Args:
        path: the path to the saved trained model
        optimier: whether to load the optimizer states (for continue training) or not (for inference)
    """

   
    loaded = torch.load(path, weights_only = False, map_location = device)
    model.load_state_dict(loaded['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(loaded['optimizer_state_dict'])
        
    if scheduler is not None:
        scheduler.load_state_dict(loaded['scheduler_state_dict'])

    return model, optimizer, scheduler
# ------------------------------------------------------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, path):
    """ Saves checkpoints while model training """

    cb_contents = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict()
    }

    # os.makedirs(path, exist_ok = True)
    torch.save(cb_contents, path)
# ------------------------------------------------------------------------------------------------------------------------
def siamese_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    train = True,
    device = config.DEVICE,
):
    """ Perform one complete training epoch """

    # cfg
    if train:
        model.train()
    else:
        model.eval()
    

    # loss tracking
    running_loss = 0.0
    total_samples = 0
    acc_distances = []
    acc_labels = []
    correct_pred = 0
    pbar = tqdm(loader, desc = 'Training' if train else 'Validation')

    with torch.set_grad_enabled(train):
        for images, labels in pbar:
            if train:
                optimizer.zero_grad()
            
            # get data
            img1 = images[0].to(device)
            img2 = images[1].to(device)
            labels = labels.to(device)
    
    
            # forward
            encodings1, encodings2 = model(img1, img2)
            loss = loss_fn(encodings1, encodings2, labels)
    
            # backward
            if train:
                loss.backward()
                optimizer.step()
    
            # track losses
            running_loss += loss.item() * labels.size(0) # batch_loss * batch_size

            # track distances/labels for acc
            d = F.pairwise_distance(encodings1, encodings2, keepdim = False)
            acc_distances.extend(d.cpu().detach().numpy())
            acc_labels.extend(labels.cpu().detach().numpy())


            # update pbar
            total_samples += labels.size(0)
            current_loss = running_loss / total_samples
            pbar.set_postfix(loss = f'{current_loss:.4f}')

            

    epoch_loss = running_loss / total_samples
            
             
    
            
    return epoch_loss, np.array(acc_distances), np.array(acc_labels)

def train_siamese(
    model,
    optimizer, 
    scheduler,
    loss_fn,
    train_data_loader,
    val_data_loader,
    epochs,
    trained_epochs,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    val_thresholds,
    best_val_acc,
    best_val_threshold,
    save_path,
    patience,
    device = config.DEVICE,
):
    """ 
    This function trains the given model for the given number of epochs (for a siamese network)

    Returns:
        results: a python dict contains:-
                    'trained_epochs': updated num of epochs trained
                    'train_losses'  : train_losses for the given epochs
                    'val_losses'    : val_losses for the given epochs
                    'val_accs'      : val_accs for the given epochs
                    'thresholds'    : thresholds for the given epochs
                    'best_acc'      : new best acc achieved
                    'best_threshold': threshold associated with the new best acc
    
    """

    model = model.to(device)


    start = trained_epochs + 1
    end = trained_epochs + epochs + 1

    counter = 0
    for epoch in range(start, end):
        print(f">> Epoch: {epoch}")

        # printing Learning Rate(s)
        for idx, param_group in enumerate(optimizer.param_groups):
            print(f'{config.BULLET} LR_{idx}: {param_group["lr"]}')
        print(f'{config.BULLET} WD: {optimizer.param_groups[0]["weight_decay"]}')
            
    
        # train
        train_loss, train_distances, train_labels = siamese_epoch(model = model, loader = train_data_loader, loss_fn = loss_fn, optimizer = optimizer)
        
        # val
        val_loss, val_distances, val_labels = siamese_epoch(model = model, loader = val_data_loader, loss_fn = loss_fn, optimizer = optimizer, train = False)

        # val acc
        val_th, val_acc = evaluation.get_best_threshold(val_distances, val_labels)

        # train acc [based on the same th]
        preds = (train_distances < val_th).astype(int)
        train_acc = np.mean(preds == train_labels)
        
    
        # print acc
        print(f'{config.BULLET} Training Acc: {train_acc}')
        print(f'{config.BULLET} Validation Acc: {val_acc}')
        print(f'{config.BULLET} Threshold: {val_th}')
        print()
        
        
        # updating
        scheduler.step(val_acc)
        trained_epochs += 1
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_thresholds.append(val_th)


        # saving best acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_threshold = val_th
            save_dir = os.path.join(save_path, f'best_results_epoch_{trained_epochs}')
            os.makedirs(save_dir, exist_ok = True)

            # saving model
            modelname = f'siamese_best_model.pth'
            training.save_checkpoint(model, optimizer, scheduler, os.path.join(save_dir, modelname))


            # saving results
            trained_results = {
                "trained_epochs" : trained_epochs,
                'train_losses'   : train_losses,
                'val_losses'     : val_losses,
                'train_accs'     : train_accs,
                'val_accs'       : val_accs,
                'val_thresholds' : val_thresholds,
                'best_val_acc'   : best_val_acc,
                'best_val_th'    : best_val_threshold 
            }

            resultsname = f'train_results.json'
            safe_results = utils.json_safe(trained_results)
            utils.save_obj(safe_results, os.path.join(save_dir, resultsname))

            counter = 0

        else:
            counter += 1

        if counter == patience:
            print("Early Stopping Activated")
            break


    return {
        "trained_epochs" : trained_epochs,
        'train_losses'   : train_losses,
        'val_losses'     : val_losses,
        'train_accs'     : train_accs,
        'val_accs'       : val_accs,
        'val_thresholds' : val_thresholds,
        'best_val_acc'   : best_val_acc,
        'best_val_th'    : best_val_threshold 
    }
# ------------------------------------------------------------------------------------------------------------------------
def plot_train_results(
    epochs: int,
    results: list,
    titles: list,
    y_labels: list
):
    """
    Plots the training results

    Args:
        epochs: number of epochs
        results: list of metric lists (train/val losses or accuracies)
        titles: titles of the plots
        y_labels: y-axis labels
    """

    x = range(1, epochs + 1)

    rows = 2 if len(results) == 4 else 1
    cols = 3 if len(results) == 3 else 2
    s = 18 if cols == 3 else 12
    h = 10 if len(results) > 2 else 6

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(s, h))
    axes = np.atleast_1d(axes).flatten()

    for res, tit, lab, ax in zip(results, titles, y_labels, axes):

        res = np.asarray(res)

        # plot
        ax.plot(x, res)
        ax.set_title(tit)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(lab)

        y_min, y_max = res.min(), res.max()
        margin = (y_max - y_min) * 0.08 if y_max != y_min else 0.01
        ax.set_ylim(y_min - margin, y_max + margin)

        min_idx = res.argmin()
        max_idx = res.argmax()

        min_x, min_y = x[min_idx], res[min_idx]
        max_x, max_y = x[max_idx], res[max_idx]

        # mark points
        ax.scatter([min_x, max_x], [min_y, max_y], zorder = 3)

        # annotate
        ax.annotate(
            f"Max {lab}: {max_y:.3f}",
            (max_x, max_y),
            xytext = (0, 10),
            textcoords="offset points",
            ha = "left",
            fontsize = 9,
            bbox = dict(boxstyle = "round, pad=0.2", fc = "white", ec = "gray")
        )

        ax.annotate(
            f"Min {lab}: {min_y:.3f}",
            (min_x, min_y),
            xytext = (0, -15),
            textcoords = "offset points",
            ha = "left",
            fontsize = 9,
            bbox = dict(boxstyle = "round,pad=0.2", fc = "white", ec = "gray")
        )

    plt.tight_layout(w_pad=3)
    plt.show()
# ------------------------------------------------------------------------------------------------------------------------
# arcface training
def arc_face_epoch(model, loader, optimizer, loss_fn, train = True, device = config.DEVICE):
    if train:
        model.train()
    else:
        model.eval()


    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    p_bar = tqdm(loader, desc = 'Training' if train else 'Validation')
    
    
    with torch.set_grad_enabled(train):
        for images, labels in p_bar:
            if train:
                optimizer.zero_grad()
                
            images, labels = images.to(device), labels.to(device)

            # forward
            if train:
                logits = model(images, labels)

            else:
                logits = model(images, labels = None) 
                
            loss = loss_fn(logits, labels)

            # backward
            if train:
                loss.backward()
                optimizer.step()

            # track loss
            running_loss += loss.item() * images.size(0)
            
            # track acc
            _, predicted = torch.max(logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += images.size(0)

            # update p_bar
            current_loss = running_loss / total_samples
            current_acc = correct_preds / total_samples
            p_bar.set_postfix(loss = f"{current_loss:.4f}", acc = f"{current_acc:.4f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    return epoch_loss, epoch_acc

def train_arc_face_model(
    model,
    train_data_loader,
    val_data_loader,
    loss_fn,
    optimizer,
    scheduler,
    epochs,
    trained_epochs,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    best_val_acc,
    device = config.DEVICE,
    patience = 10,
    save_path = "/kaggle/working/"
):
    """
    Args:
        model: ArcFace Model
        train_data_loader, val_data_loader: data loaders, should sample (images, labels)
        loss_fn, optimizer, scheduler: train configurations
        epochs: num of epochs for this run
        trained_epochs: num of trained epochs before this run
        train_losses, val_losses, train_accs, val_accs: training results before this run
        best_val_acc: best_val_acc before this run
        device: device to train the model on
        patience: for early stopping
        save_path: the path to save the model if achieved acc > best_val_acc

    Returns:
        results: a python dict contains (trained_epochs, train_losses, val_losses, train_accs, val_accs, best_val_acc)
    """
    model = model.to(device)

    start = trained_epochs + 1
    end = trained_epochs + epochs + 1

    counter = 0
    for epoch in range(start, end):
        print(f">> Epoch: {epoch}")
        
        for idx, param_group in enumerate(optimizer.param_groups):
            print(f'{config.BULLET} LR_{idx}: {param_group["lr"]}')
        print(f'{config.BULLET} WD: {optimizer.param_groups[0]["weight_decay"]}')

        # train
        train_loss, train_acc = arc_face_epoch(model, train_data_loader, optimizer, loss_fn)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # val
        val_loss, val_acc = arc_face_epoch(model, val_data_loader, optimizer, loss_fn, train = False)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # update
        trained_epochs += 1
        scheduler.step(val_acc)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # saving model
            model_path = os.path.join(save_path, "arc_face_best_model.pth")
            training.save_checkpoint(model, optimizer, scheduler, model_path)

            # saving results
            trained_results = {
                "trained_epochs" :trained_epochs,
                'train_losses'   :train_losses,
                'val_losses'     :val_losses,
                'train_accs'     :train_accs,
                'val_accs'       :val_accs,
                'best_val_acc'   :best_val_acc
            }
            
            results_path = os.path.join(save_path, f'arcface5_{trained_epochs}_epochs_results.json')
            safe_results = utils.json_safe(trained_results)
            utils.save_obj(safe_results, results_path)

            counter = 0

        else:
            counter += 1


        if counter == patience:
            print("Early Stopping Activated")
            break

    return {
        "trained_epochs" :trained_epochs,
        'train_losses'   :train_losses,
        'val_losses'     :val_losses,
        'train_accs'     :train_accs,
        'val_accs'       :val_accs,
        'best_val_acc'   :best_val_acc
    }
# ------------------------------------------------------------------------------------------------------------------------
# I-ResNet
def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride=stride, padding=1, bias = False)
    
# def iresnet50(): return IResNet(IBasicBlock, [3, 4, 14, 3])