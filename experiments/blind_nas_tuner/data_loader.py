import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def get_dataloaders(n_samples=5000, n_features=100, n_classes=5, batch_size=64):
    """
    Creates an obscured synthetic dataset to prevent the LLM 
    from relying on pre-training bias (e.g., memorizing MNIST).
    """
    # Create a complex, nonlinear classification problem
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=30, 
        n_redundant=20,
        n_classes=n_classes, 
        random_state=42
    )
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, n_features, n_classes
