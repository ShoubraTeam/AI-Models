import pandas as pd
import numpy as np
import torch
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import src.config as config
from src.model import ProfileMLP

test_cases = [
    {
        "name": "The Confused Veteran",
        "role_category": "AI Engineer",
        "bio": "I make logos.",
        "skills": "Photoshop",
        "num_projects": 50, "hourly_rate": 30, "num_certifications": 0,
        "time_to_respond": 24, "last_active": 30, 
        "is_professional_photo": 0, 
        "mentorship_offered": 0
    },
    {
        "name": "The Perfect Newbie",
        "role_category": "AI Engineer",
        "bio": "Python Expert.",
        "skills": "Python, PyTorch",
        "num_projects": 0, "hourly_rate": 50, "num_certifications": 2,
        "time_to_respond": 1, "last_active": 1, 
        "is_professional_photo": 1, 
        "mentorship_offered": 0
    }
]

class InferencePipeline:
    def __init__(self):
        self.cfg = config.MODEL_CONFIGS[config.ACTIVE_MODEL_NAME]
        self.device = config.DEVICE
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.scaler = StandardScaler()
        
        self.expected_cat_cols = [
            'is_professional_photo_1.0',  
            'mentorship_offered_1.0'
        ]
        print(f">>> Initializing Pipeline with Dynamic Column Filling.")
        self._fit_preprocessors()

    def _apply_engineering(self, df, fit_vectorizer=False):
        temp_df = df.copy()
        
        temp_df['text_content'] = temp_df['bio'].fillna('') + " " + temp_df['skills'].fillna('')
        temp_df['role_content'] = temp_df['role_category'].fillna('')

        if fit_vectorizer:
            all_text = pd.concat([temp_df['text_content'], temp_df['role_content']])
            self.vectorizer.fit(all_text)

        try:
            prof_vec = self.vectorizer.transform(temp_df['text_content'])
            role_vec = self.vectorizer.transform(temp_df['role_content'])
            sims = []
            for i in range(len(temp_df)):
                if prof_vec[i].nnz == 0 or role_vec[i].nnz == 0:
                    sims.append(0.0)
                else:
                    sim = cosine_similarity(prof_vec[i], role_vec[i])[0][0]
                    sims.append(sim)
            temp_df['role_consistency_score'] = sims
        except:
            temp_df['role_consistency_score'] = 0.0

        projects = temp_df.get('num_projects', 0)
        rate = temp_df.get('hourly_rate', 0)
        
        temp_df['projects_weighted_by_consistency'] = projects * temp_df['role_consistency_score']
        temp_df['rate_weighted_by_consistency'] = rate * temp_df['role_consistency_score']
        
        return temp_df

    def _get_feature_lists(self):
        num_features = [
            'num_certifications', 'time_to_respond', 'last_active',
            'role_consistency_score', 'projects_weighted_by_consistency', 'rate_weighted_by_consistency'
        ]
        cat_features = list(config.CATEGORICAL_FEATURES)
        return num_features, cat_features

    def _fit_preprocessors(self):
        if os.path.exists(config.ROOT_DATA_DIR):
            df = pd.read_csv(config.ROOT_DATA_DIR)
        else:
            df = pd.read_csv("data/freelancers_dataset_updated.csv")

        cols_to_drop = ['num_reviews', 'avg_rating']
        df_clean = df.drop(columns=cols_to_drop, errors='ignore')
        
        df_processed = self._apply_engineering(df_clean, fit_vectorizer=True)
        self.final_num_features, self.final_cat_features = self._get_feature_lists()
        
        self.scaler.fit(df_processed[self.final_num_features].fillna(0))

    def preprocess_new_data(self, new_data_list):
        df_new = pd.DataFrame(new_data_list)
        df_processed = self._apply_engineering(df_new, fit_vectorizer=False)
        
        X_num = df_processed[self.final_num_features].fillna(0)
        X_num_scaled = self.scaler.transform(X_num)
        
        X_cat = df_processed[self.final_cat_features]
        
        X_cat_encoded = pd.get_dummies(X_cat, columns=self.final_cat_features, drop_first=True)
                
        for col_name in self.final_cat_features:
            found = False
            for col in X_cat_encoded.columns:
                if col.startswith(col_name) and (col.endswith('_1') or col.endswith('_1.0')):
                    found = True
                    break
            
            if not found:
                missing_col_name = f"{col_name}_1" 
                X_cat_encoded[missing_col_name] = 0

        
        final_cat_matrix = np.zeros((len(df_new), 2)) 
        
        
        photo_cols = [c for c in X_cat_encoded.columns if 'is_professional_photo' in c]
        if photo_cols:
            final_cat_matrix[:, 0] = X_cat_encoded[photo_cols[0]].values
            
        mentor_cols = [c for c in X_cat_encoded.columns if 'mentorship_offered' in c]
        if mentor_cols:
            final_cat_matrix[:, 1] = X_cat_encoded[mentor_cols[0]].values

        X_final = np.hstack([X_num_scaled, final_cat_matrix])
        
        return X_final

    def _find_model_file_smart(self, save_dir, target_name):
        if not os.path.exists(save_dir): return None
        clean_target = target_name.strip().lower()
        files = os.listdir(save_dir)
        for f in files:
            if "_logs" in f or "_preprop" in f: continue
            if not f.endswith(('.joblib', '.pth')): continue
            if clean_target in f.lower():
                return os.path.join(save_dir, f)
        return None

    def predict(self, data, specific_model_name=None):
        X_processed = self.preprocess_new_data(data)
        save_dir = os.path.abspath(config.MODEL_SAVE_PATH)
        
        target_name = specific_model_name if specific_model_name else config.ACTIVE_MODEL_NAME
        model_path = self._find_model_file_smart(save_dir, target_name)
        
        if not model_path:
            raise FileNotFoundError(f"Model '{target_name}' not found.")
        
        print(f"    >>> Loading: {os.path.basename(model_path)}")
        
        if self.cfg["framework"] == "sklearn":
            loaded_obj = joblib.load(model_path)
            model = loaded_obj
            if isinstance(loaded_obj, dict):
                model = list(loaded_obj.values())[0]
            return model.predict(X_processed)
            
        elif self.cfg["framework"] == "pytorch":
            hidden_layers = self.cfg.get("hidden_layers", [64, 32, 16])
            model = ProfileMLP(input_dim=8, hidden_layers=hidden_layers)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
                return model(X_tensor).cpu().numpy().flatten()


if __name__ == "__main__":
    import src.config as config  
    
    print("="*60)
    print("STARTING GLOBAL INFERENCE BENCHMARK (DYNAMIC FILL)")
    print("="*60)

    all_config_keys = list(config.MODEL_CONFIGS.keys())

    for config_key in all_config_keys:
        print(f"\nConfiguration Group: {config_key.upper()}")
        print("-" * 40)
        
        config.ACTIVE_MODEL_NAME = config_key
        current_cfg = config.MODEL_CONFIGS[config_key]
        
        try:
            pipeline = InferencePipeline()
            model_types = current_cfg.get("model_type")
            models_to_run = model_types if isinstance(model_types, list) else [config_key] 

            for model_name in models_to_run:
                print(f"Testing Algorithm: {model_name}")
                try:
                    preds = pipeline.predict(test_cases, specific_model_name=model_name)
                    print(f"Results for {model_name}:")
                    for i, case in enumerate(test_cases):
                        print(f"  {case['name']:<35} -> Score: {preds[i]:.2f}")
                except Exception as e:
                    print(f"Runtime Error for {model_name}: {e}")

        except Exception as e:
            print(f"Critical Error initializing pipeline for {config_key}: {e}")
            
    print("\n" + "="*60)
    print("Benchmark Completed")
    print("="*60)