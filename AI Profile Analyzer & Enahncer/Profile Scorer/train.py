import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader
import src.config as config
from src.model import get_model
from src.dataset import get_loaders, ProfileDataset
from src.utils import EarlyStopping
from src.engine import train_one_epoch, evaluate

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_optimizer(model, cfg):
    lr = cfg.get("lr", 0.001)
    opt_name = cfg.get("optimizer", "Adam")
    
    if opt_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr)
    else:
        # Default fallback
        print(f"Unknown optimizer '{opt_name}', defaulting to Adam.")
        return optim.Adam(model.parameters(), lr=lr)

def get_criterion(cfg):
    loss_name = cfg.get("loss", "MSE")    
    if loss_name == "MSE":
        return nn.MSELoss()
    elif loss_name == "L1": # MAE Loss
        return nn.L1Loss()
    elif loss_name == "SmoothL1":
        return nn.SmoothL1Loss()
    else:
        print(f"Unknown loss '{loss_name}', defaulting to MSELoss.")
        return nn.MSELoss()

def run_train():
    cfg = config.MODEL_CONFIGS[config.ACTIVE_MODEL_NAME]
    
    X_train, X_test, y_train, y_test = get_loaders(cfg)

    if cfg["framework"] == "sklearn":
        print(f"[Framework: Sklearn] Experiment: {config.ACTIVE_MODEL_NAME}")
        
        models_output = get_model(config.ACTIVE_MODEL_NAME)
        
        if isinstance(models_output, dict):
            models_dict = models_output
        else:
            models_dict = {config.ACTIVE_MODEL_NAME: models_output}
            
        metrics_summary = []

        for name, model in models_dict.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics_summary.append({'Model': name, 'MAE': mae, 'R2': r2})
            print(f"{name} | MAE: {mae:.4f} | R2: {r2:.4f}")
            
            joblib.dump(model, f"{config.MODEL_SAVE_PATH}/{name}.joblib")
            
        print("\nFINAL COMPARISON TABLE:")
        print(pd.DataFrame(metrics_summary))

    else:
        print(f"[Framework: PyTorch] Model: {config.ACTIVE_MODEL_NAME}")
        
        input_dim = X_train.shape[1]
        print(f"Input Dimension detected: {input_dim}")
        
        model = get_model(config.ACTIVE_MODEL_NAME, input_dim=input_dim)
        model.to(config.DEVICE)
        
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)

        train_dataset = ProfileDataset(X_train, y_train)
        test_dataset = ProfileDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)
        
        scheduler = None
        if cfg.get("use_scheduler", False):
            print("Scheduler Activated (ReduceLROnPlateau)")
            patience = cfg.get("scheduler_patience", config.LR_PATIENCE) if hasattr(config, 'LR_PATIENCE') else 3
            factor = cfg.get("lr_factor", config.LR_FACTOR) if hasattr(config, 'LR_FACTOR') else 0.1
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=factor, patience=patience
            )
        else:
            print("Scheduler Deactivated")

        best_mae = float('inf')
        early_stopping = EarlyStopping(
            patience=cfg.get("patience", 10),
            save_path=f"{config.MODEL_SAVE_PATH}/{config.ACTIVE_MODEL_NAME}_best_mae.pth"
        )
        
        results_log = []

        for epoch in range(cfg["epochs"]):
            avg_train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
            
            val_loss, _ = evaluate(model, test_loader, criterion, config.DEVICE)
            
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(config.DEVICE)
                y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(config.DEVICE).view(-1, 1)
                preds = model(X_test_tensor)
                current_mae = mean_absolute_error(y_test_tensor.cpu().numpy(), preds.cpu().numpy())

            if scheduler:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{cfg['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {current_mae:.4f} | LR: {current_lr:.6f}")

            results_log.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_mae": current_mae,
                "lr": current_lr
            })
            df_log = pd.DataFrame(results_log)
            log_path = os.path.join(config.MODEL_SAVE_PATH, f"{config.ACTIVE_MODEL_NAME}_logs.csv")
            df_log.to_csv(log_path, index=False)
            early_stopping(val_loss, model)
            
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
            if current_mae < best_mae:
                best_mae = current_mae
                torch.save(model.state_dict(), f"{config.MODEL_SAVE_PATH}/{config.ACTIVE_MODEL_NAME}_best_mae.pth")
        
        print(f"Training Complete. Best MAE: {best_mae:.4f}")
        print(f"Logs saved to: {config.MODEL_SAVE_PATH}/{config.ACTIVE_MODEL_NAME}_logs.csv")

if __name__ == "__main__":
    run_train()