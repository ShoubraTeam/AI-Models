import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import src.config as config

class ProfileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def apply_feature_engineering(df):
    temp_df = df.copy()
    
    temp_df['text_content'] = temp_df['bio'].fillna('') + " " + temp_df['skills'].fillna('')
    temp_df['role_content'] = temp_df['role_category'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    all_text = pd.concat([temp_df['text_content'], temp_df['role_content']])
    tfidf.fit(all_text)

    prof_vec = tfidf.transform(temp_df['text_content'])
    role_vec = tfidf.transform(temp_df['role_content'])

    sims = []
    for i in range(len(temp_df)):
        sim = cosine_similarity(prof_vec[i], role_vec[i])[0][0]
        sims.append(sim)
    
    temp_df['role_consistency_score'] = sims

    temp_df['projects_weighted_by_consistency'] = temp_df['num_projects'] * temp_df['role_consistency_score']
    temp_df['rate_weighted_by_consistency'] = temp_df['hourly_rate'] * temp_df['role_consistency_score']
    
    return temp_df

def get_loaders(cfg):
    df = pd.read_csv(config.ROOT_DATA_DIR)
    cols_to_drop = ['num_reviews', 'avg_rating']
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')
    
    df_processed = apply_feature_engineering(df_clean)
    
    num_features = list(config.NUMERIC_FEATURES)
    cat_features = list(config.CATEGORICAL_FEATURES)
    
    if cfg.get("add_consistency_score") or cfg.get("add_interaction_features"):
        num_features += ['role_consistency_score', 'projects_weighted_by_consistency', 'rate_weighted_by_consistency']
        if 'num_projects' in num_features: num_features.remove('num_projects')
        if 'hourly_rate' in num_features: num_features.remove('hourly_rate')
    
    X_num = df_processed[num_features]
    X_cat = df_processed[cat_features]
    y = df_processed[config.TARGET]

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    X_cat_encoded = pd.get_dummies(X_cat, columns=cat_features, drop_first=True, dtype=float)

    X_final = np.hstack([X_num_scaled, X_cat_encoded.values])

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=config.TEST_SPLIT, random_state=42
    )
    
    return X_train, X_test, y_train, y_test