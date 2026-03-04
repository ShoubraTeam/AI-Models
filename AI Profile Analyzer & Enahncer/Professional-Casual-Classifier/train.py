import torch.nn as nn
import torch.optim as optim
import src.config as config
from src.model import get_model
from src.dataset import get_loaders, prepare_dataset
from src.engine import train_one_epoch, evaluate
import pandas as pd
from src.utils import EarlyStopping 
import os 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
prepare_dataset(config.ROOT_DATA_DIR)

def get_optimizer(model, cfg):
    if cfg["optimizer"] == "Adam":
        return optim.Adam(model.parameters(), lr=cfg["lr"])
    elif cfg["optimizer"] == "SGD":
        return optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    elif cfg["optimizer"] == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=cfg["lr"])
    else:
        raise ValueError(f"Unknown Optimizer: {cfg['optimizer']}")

def get_criterion(cfg):
    if cfg["loss"] == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif cfg["loss"] == "BCELoss":
        return nn.BCELoss()
    else:
        raise ValueError(f"Unknown Loss Function: {cfg['loss']}")

def run():
    cfg = config.CURRENT_CONFIG
    
    print(f"Starting Experiment: {cfg['name']}")
    print(f"Settings: Opt={cfg['optimizer']}, Loss={cfg['loss']}, LR={cfg['lr']}")
    # 1. Load Data & Model
    train_loader, val_loader = get_loaders(cfg)
    model = get_model(cfg["name"]).to(config.DEVICE)
    
    # 2. Get Optimizer & Loss
    optimizer = get_optimizer(model, cfg)
    criterion = get_criterion(cfg)

    scheduler = None
    if cfg.get("use_scheduler", False): 
        print("Scheduler Activated (ReduceLROnPlateau)")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.LR_FACTOR, patience=config.LR_PATIENCE, min_lr=config.MIN_LR
        )
    else:
        print("Scheduler Deactivated ( relying on Optimizer )")

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        save_path=f"{config.MODEL_SAVE_PATH}/{cfg['name']}_best.pth"
    )

    results = []
    
    for epoch in range(cfg["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, config.DEVICE
        )
        
        print(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"   Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
  
        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        results.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr 
        })
        df = pd.DataFrame(results)
        df.to_csv(f"{config.MODEL_SAVE_PATH}/{cfg['name']}_logs.csv", index=False)

        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered! Model has stopped improving.")
            break
            
    print("Training Complete!")

if __name__ == "__main__":
    run()