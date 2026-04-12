"""
AITL V2 — Model-Agnostic Trainer
Executes ANY Python code the agent generates (sklearn, PyTorch, raw numpy, etc.)
Measures results externally — agent can't cheat the metric.
"""
import traceback
import signal
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelBinarizer

TIMEOUT_SECONDS = 120  # Safety cap per iteration


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError(f"Code execution exceeded {TIMEOUT_SECONDS} seconds")


def execute_agent_code(code_str, X_train, y_train, X_val, y_val, n_classes, timeout=TIMEOUT_SECONDS):
    """
    Execute agent-generated code and measure results.
    
    The agent must define:
        def solve(X_train, y_train, X_val, y_val):
            # ... any code ...
            return predictions  # numpy array, shape (n_val,)
    
    Returns:
        (results_dict, None) on success
        (None, error_string) on failure
    """
    # Build execution namespace with common ML libraries
    exec_namespace = {
        '__builtins__': __builtins__,
        'np': np,
        'numpy': np,
    }
    
    # Inject libraries that are available
    try:
        import sklearn
        exec_namespace['sklearn'] = sklearn
        # Pre-import common sklearn submodules
        from sklearn import ensemble, linear_model, svm, tree, neighbors
        from sklearn import preprocessing, pipeline, model_selection
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.svm import SVC, LinearSVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        from sklearn.pipeline import Pipeline
        exec_namespace.update({
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'AdaBoostClassifier': AdaBoostClassifier,
            'ExtraTreesClassifier': ExtraTreesClassifier,
            'LogisticRegression': LogisticRegression,
            'SGDClassifier': SGDClassifier,
            'SVC': SVC,
            'LinearSVC': LinearSVC,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'KNeighborsClassifier': KNeighborsClassifier,
            'MLPClassifier': MLPClassifier,
            'StandardScaler': StandardScaler,
            'MinMaxScaler': MinMaxScaler,
            'Pipeline': Pipeline,
        })
    except ImportError:
        pass
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        exec_namespace.update({
            'torch': torch, 'nn': nn, 'F': F, 'optim': optim,
        })
    except ImportError:
        pass

    try:
        # Execute the agent's code (defines solve())
        exec(code_str, exec_namespace)
        
        solve_fn = exec_namespace.get('solve')
        if not solve_fn:
            return None, "Structure Error: Must define 'def solve(X_train, y_train, X_val, y_val)' that returns predictions."
        
        # Run with timeout (Unix only; on Windows, use threading)
        import threading
        result_holder = [None]
        error_holder = [None]
        
        def run_solve():
            try:
                result_holder[0] = solve_fn(
                    X_train.copy(), y_train.copy(), 
                    X_val.copy(), y_val.copy()
                )
            except Exception as e:
                error_holder[0] = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
        
        thread = threading.Thread(target=run_solve)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            return None, f"Timeout Error: Code execution exceeded {timeout} seconds. Try a simpler/faster model."
        
        if error_holder[0]:
            return None, error_holder[0]
        
        predictions = result_holder[0]
        
        # Validate predictions
        if predictions is None:
            return None, "Error: solve() returned None. Must return predictions array."
        
        predictions = np.array(predictions).flatten()
        
        if len(predictions) != len(y_val):
            return None, f"Shape Error: Expected {len(y_val)} predictions, got {len(predictions)}."
        
        # Ensure predictions are integer class labels
        predictions = predictions.astype(int)
        
        # Validate prediction range
        unique_preds = np.unique(predictions)
        if np.any(unique_preds < 0) or np.any(unique_preds >= n_classes):
            return None, f"Range Error: Predictions must be in [0, {n_classes-1}], got range [{unique_preds.min()}, {unique_preds.max()}]."
        
        # --- External measurement (agent can't influence this) ---
        acc = accuracy_score(y_val, predictions)
        
        # Compute log-loss from predictions (convert to one-hot probabilities)
        lb = LabelBinarizer()
        lb.fit(range(n_classes))
        pred_onehot = lb.transform(predictions)
        # Add small epsilon to avoid log(0)
        pred_proba = pred_onehot * 0.95 + (1 - pred_onehot) * (0.05 / (n_classes - 1))
        loss = log_loss(y_val, pred_proba)
        
        return {
            "val_accuracy": round(acc, 4),
            "val_loss": round(loss, 4),
            "n_unique_predictions": len(unique_preds),
            "prediction_distribution": np.bincount(predictions, minlength=n_classes).tolist(),
        }, None
        
    except Exception as e:
        return None, f"Execution Error: {str(e)}\n{traceback.format_exc()}"
