import os
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import src.config as config
from src.model import ProfileMLP
from src.dataset import get_loaders


def find_model_file(save_dir, model_name_identifier):
    abs_save_dir = os.path.abspath(save_dir)
    
    if not os.path.exists(abs_save_dir):
        print(f"Error: Save directory does not exist: {abs_save_dir}")
        return None

    clean_identifier = model_name_identifier.strip()
    
    files = os.listdir(abs_save_dir)
    
    for f in files:
        if "_logs" in f or "_preprop" in f:
            continue
        if not f.endswith(('.pth', '.joblib')):
            continue
            
        if clean_identifier.lower() in f.lower():
            return os.path.join(abs_save_dir, f)
            
    return None

def run_evaluation():
    save_dir = os.path.abspath(config.MODEL_SAVE_PATH)
    device = config.DEVICE
    results_table = []

    print("="*60)
    print(f"STARTING EVALUATION BENCHMARK")
    print(f"Looking for models in: {save_dir}")
    print("="*60)

    for config_key, cfg in config.MODEL_CONFIGS.items():
        print(f"\n>> Evaluating Configuration: {config_key}")
        
        try:
            _, X_test, _, y_test = get_loaders(cfg)
            y_true = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        except Exception as e:
            print(f"Skipping data load for {config_key}: {e}")
            continue

        target_models = []
        if isinstance(cfg["model_type"], list):
            target_models = cfg["model_type"]
        else:
            target_models = [config_key]

        for model_name in target_models:
            clean_name = model_name.strip()

            model_path = find_model_file(save_dir, clean_name)
            
            if not model_path:
                print(f"Model file NOT found for: '{clean_name}'")
                continue
            
            print(f"Loaded: {os.path.basename(model_path)}")
            
            try:
                if cfg["framework"] == "sklearn":
                    loaded_obj = joblib.load(model_path)
                    
                    if isinstance(loaded_obj, dict):
                        found_key = None
                        for k in loaded_obj.keys():
                            if clean_name.lower() in k.lower():
                                found_key = k
                                break
                        if found_key:
                            model = loaded_obj[found_key]
                        else:
                            model = list(loaded_obj.values())[0]
                    else:
                        model = loaded_obj
                    
                    y_pred = model.predict(X_test)

                elif cfg["framework"] == "pytorch":
                    input_dim = X_test.shape[1]
                    hidden_layers = cfg.get("hidden_layers", [64, 32, 16])
                    
                    model = ProfileMLP(input_dim=input_dim, hidden_layers=hidden_layers)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.to(device)
                    model.eval()
                    
                    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        y_pred = model(X_tensor).cpu().numpy().flatten()

                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                results_table.append({'Model': clean_name, 'Config': config_key, 'MAE': mae, 'R2': r2})
                print(f"      -> MAE: {mae:.4f} | R2: {r2:.4f}")

                plot_dir = os.path.join(os.getcwd(), "plots_results")
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                
                plt.figure(figsize=(5, 3))
                sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
                plt.title(f'{clean_name} (MAE: {mae:.2f})')
                plt.xlabel('True Score')
                plt.ylabel('Predicted Score')
                
                filename = f"{clean_name}_prediction_plot.png"
                plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight', dpi=300)
                plt.close()

            except Exception as e:
                print(f"CRITICAL ERROR evaluating {clean_name}: {e}")

    if results_table:
        print("\n" + "="*60)
        print("GLOBAL MODELS LEADERBOARD")
        print("="*60)
        df_results = pd.DataFrame(results_table).sort_values(by='MAE')
        print(df_results[['Model', 'Config', 'MAE', 'R2']].to_string(index=False))
    else:
        print("No models were evaluated successfully.")

if __name__ == "__main__":
    run_evaluation()