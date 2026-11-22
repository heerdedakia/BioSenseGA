import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import FEATURE_NAMES

def preprocess_data(df, test_size=0.2, random_state=42):
    feature_cols = [f'feature_{i}' for i in range(1, 129)]
    X = df[feature_cols].copy()
    y = df['class_label'].copy()
    
    # Handle missing values by filling with column mean
    X = X.fillna(X.mean())
    
    # Stratified split - preserves index to retrieve batch_id and metadata
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Retain raw unscaled data and indices for genuine sample extraction
    test_indices = X_test.index.tolist()
    X_test_raw = X_test.values.copy()
    X_train_raw = X_train.values.copy()
    
    test_meta = df.loc[test_indices, ['class_label', 'batch_id']].copy()
    
    # Fit StandardScaler STRICTLY on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Encode labels (e.g. 1..6 -> 0..5)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'test_indices': test_indices,
        'test_meta': test_meta,
        'y_train': y_train_encoded,
        'y_test': y_test_encoded,
        'y_train_raw': y_train.values,
        'y_test_raw': y_test.values,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': FEATURE_NAMES,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test))
    }
