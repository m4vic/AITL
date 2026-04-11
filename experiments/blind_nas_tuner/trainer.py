import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_and_evaluate(model_code, train_loader, val_loader, max_epochs=30, epoch_patience=3):
    """
    Executes LLM-generated PyTorch code with dynamic epoch stopping.
    
    Instead of a fixed epoch count, training continues epoch-by-epoch and 
    stops automatically when the per-epoch validation loss stops improving.
    This implements a nested AITL loop: the trainer itself is self-evaluating.
    
    Args:
        max_epochs: Hard upper limit on epochs (safety cap)
        epoch_patience: Stop training if loss doesn't improve for this many consecutive epochs
    
    Returns:
        results dict with metrics + epoch_history, or (None, error_str)
    """
    # Use a SINGLE namespace dict for exec so class and function defs see each other
    exec_namespace = {
        'torch': torch,
        'nn': nn,
        'F': F,
        'optim': optim,
        '__builtins__': __builtins__,
    }
    
    try:
        exec(model_code, exec_namespace)
        
        model_class = exec_namespace.get('Model')
        get_optimizer = exec_namespace.get('get_optimizer')
        
        if not model_class or not get_optimizer:
            return None, "Structure Error: Must define 'class Model(nn.Module)' and 'def get_optimizer(model)'."
            
        model = model_class()
        optimizer = get_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        
        epoch_history = []       # (epoch, train_loss, val_loss, val_acc)
        best_val_loss = float('inf')
        no_improve_count = 0
        epochs_run = 0
        
        print(f"    [Trainer] Starting dynamic training (max {max_epochs} epochs, patience={epoch_patience})...")
        
        for epoch in range(1, max_epochs + 1):
            epochs_run = epoch
            
            # --- Training pass ---
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            
            # --- Validation pass ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            epoch_history.append((epoch, avg_train_loss, avg_val_loss, val_acc))
            
            print(f"    [Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")
            
            # --- Per-epoch early stopping check ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                no_improve_count = 0
                print(" ✓")  # Improvement marker
            else:
                no_improve_count += 1
                print(f" [no improve {no_improve_count}/{epoch_patience}]")
                
                if no_improve_count >= epoch_patience:
                    print(f"    [Trainer] Early stopping at epoch {epoch} — loss not improving.")
                    break
                    
        print(f"    [Trainer] Finished: {epochs_run} epochs run. Best Val Loss: {best_val_loss:.4f}")
        
        return {
            "final_val_loss": best_val_loss,
            "val_acc": best_val_acc,
            "epochs_run": epochs_run,
            "epoch_history": epoch_history,
        }, None
        
    except Exception as e:
        return None, f"Execution Error: {str(e)}\n{traceback.format_exc()}"
