import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

import src.config as config
from src.model import get_model
from src.dataset import get_loaders

def run_evaluation():
    cfg = config.CURRENT_CONFIG
    device = config.DEVICE

    _, test_loader = get_loaders(cfg)
    
    model = get_model(cfg["name"]).to(device)
    
    model_path = f"{config.MODEL_SAVE_PATH}/{cfg['name']}_best.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from: {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Please train first.")
        return

    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running Predictions on Test Set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            preds = torch.round(torch.sigmoid(outputs))
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    print("Classification Report:")
    target_names = ['Casual', 'Professional'] 
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    
    save_path = f"{config.MODEL_SAVE_PATH}/{cfg['name']}_cm.png"
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_evaluation()