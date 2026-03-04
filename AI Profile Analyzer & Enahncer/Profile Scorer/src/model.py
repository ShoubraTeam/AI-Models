import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import src.config as config


class ProfileMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32, 16]):
        super(ProfileMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_layers[0])
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.output = nn.Linear(hidden_layers[2], 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        return self.output(x)

def get_model(model_name, input_dim=None):
    cfg = config.MODEL_CONFIGS[model_name]
    
    if cfg["framework"] == "sklearn":
        if isinstance(cfg["model_type"], list):
            models = {}
            for m_type in cfg["model_type"]:
                m_params = cfg.get("params", {}).get(m_type, {})
                
                if m_type == "LinearRegression": model = LinearRegression(**m_params)
                elif m_type == "Ridge": model = Ridge(**m_params)
                elif m_type == "RandomForestRegressor": model = RandomForestRegressor(**m_params)
                elif m_type == "GradientBoostingRegressor": model = GradientBoostingRegressor(**m_params)
                else: continue
                
                models[m_type] = model
            return models
        
        else:
            m_type = cfg["model_type"]
            m_params = cfg.get("params", {}) # Params are usually direct for single models
            
            if m_type == "LinearRegression": return LinearRegression(**m_params)
            elif m_type == "Ridge": return Ridge(**m_params)
            elif m_type == "RandomForestRegressor": return RandomForestRegressor(**m_params)
            elif m_type == "GradientBoostingRegressor": return GradientBoostingRegressor(**m_params)
            else: return LinearRegression(**m_params)

    else:
        if input_dim is None:
            input_dim = 132 
            
        hidden_layers = cfg.get("hidden_layers", [64, 32, 16])
        model = ProfileMLP(input_dim, hidden_layers)
        return model