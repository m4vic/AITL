"""
AITL V2 — Data Loader
Uses a REAL dataset (Cover Type) with all labels stripped to numbers.
The agent never knows what the data represents.

NOTE: Data is passed RAW (no preprocessing). The agent decides its
own preprocessing (scaling, PCA, etc.) inside solve().
"""
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

def get_data(n_samples=10000, seed=42):
    """
    Load Cover Type dataset, subsample, strip all labels.
    
    Returns:
        X_train, y_train, X_val, y_val: numpy arrays
        n_features: int
        n_classes: int
    """
    print("  [Data] Downloading/loading Cover Type dataset...")
    data = fetch_covtype()
    X_full, y_full = data.data, data.target
    
    # Cover Type classes are 1-7, remap to 0-6
    y_full = y_full - 1
    
    # Stratified subsample for speed
    if n_samples < len(X_full):
        X_full, _, y_full, _ = train_test_split(
            X_full, y_full, train_size=n_samples, 
            random_state=seed, stratify=y_full
        )
    
    # Train/val split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full
    )
    
    # NOTE: No preprocessing applied. Agent decides its own scaling/normalization.
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    print(f"  [Data] Loaded: {X_train.shape[0]} train, {X_val.shape[0]} val")
    print(f"  [Data] Features: {n_features}, Classes: {n_classes}")
    print(f"  [Data] Class distribution: {np.bincount(y_train.astype(int))}")
    print(f"  [Data] Agent sees: n_features={n_features}, n_classes={n_classes}")
    print(f"  [Data] Agent does NOT know: dataset name, feature names, class meanings")
    
    return X_train, y_train.astype(int), X_val, y_val.astype(int), n_features, n_classes
