import torch
from PIL import Image
import src.config as config
from src.model import get_model
from src.dataset import get_transforms
import argparse 

def predict_image(image_path, model_path=None):
    cfg = config.CURRENT_CONFIG
    device = config.DEVICE
    
    print(f"Loading model architecture: {cfg['name']}...")
    model = get_model(cfg["name"])
    
    if model_path is None:
        model_path = r"models\baseline_cnn_best.pth"
        
    print(f"Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() 
    
    transform = get_transforms(cfg)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
    
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        
        label = "Professional" if prob > 0.5 else "Casual"
        confidence = prob if prob > 0.5 else 1 - prob
        
    print(f"Image: {image_path}")
    print(f"Prediction: {label} ({confidence*100:.2f}%)")
    
    return label, confidence

if __name__ == "__main__":
    
    predict_image(r"data\test\casual\img_13.jpg")