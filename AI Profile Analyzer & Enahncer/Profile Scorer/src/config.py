import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "models/" 
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

if os.path.exists("/kaggle/input"):
    ROOT_DATA_DIR = "/kaggle/input/professional-casual-dataset" 
    
else:
    ROOT_DATA_DIR = "data/freelancers_dataset_updated.csv"


MODEL_CONFIGS = {
    "linear_reg": {
        "framework": "sklearn",
        "model_type": "LinearRegression",
        "use_text": "tfidf",
        "add_consistency_score": True
    },
    
    "ml_consistency_group": {
        "framework": "sklearn",
        "add_consistency_score": True, 
        "use_text": "tfidf",
        "model_type": [
            "Ridge",
            "RandomForestRegressor",
            "GradientBoostingRegressor"
        ],
        "params": {
            "RandomForestRegressor": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "GradientBoostingRegressor": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
        }
    },
    
    "MLP": {
        "framework": "pytorch",
        "model_type": "MLP",
        "epochs": 50,
        "batch_size": 32,
        "lr": 0.001,
        "optimizer": "adam",
        "hidden_layers": [64, 32, 16],
        "use_text": "tfidf",
        "add_interaction_features": True,
        "use_scheduler": True,
        "loss": "L1"
    }
}

PATIENCE = 15       
LR_FACTOR = 0.5      
LR_PATIENCE = 4      
MIN_LR = 1e-7

NUMERIC_FEATURES = ['hourly_rate', 'num_certifications', 'time_to_respond', 'last_active', 'num_projects']
CATEGORICAL_FEATURES = ['is_professional_photo', 'mentorship_offered']

TEXT_FEATURES = {'bio': 50, 'skills': 50}

COMBINED_TEXT_COL = 'text_content' 
ROLE_TEXT_COL = 'role_content'

TARGET = 'profile_score'

TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

ACTIVE_MODEL_NAME = "MLP" 
