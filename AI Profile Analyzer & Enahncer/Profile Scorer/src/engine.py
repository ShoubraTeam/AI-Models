import torch
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_mae = 0.0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for inputs, labels in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        mae = torch.abs(outputs.detach() - labels).mean().item()
        total_mae += mae
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    avg_mae = total_mae / len(loader)
    
    return avg_loss, avg_mae

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            mae = torch.abs(outputs - labels).mean().item()
            total_mae += mae
            
    avg_loss = running_loss / len(loader)
    avg_mae = total_mae / len(loader)
    
    return avg_loss, avg_mae