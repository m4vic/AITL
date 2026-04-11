import os
import json
import datetime
import matplotlib.pyplot as plt
from data_loader import get_dataloaders
from trainer import train_and_evaluate
from agent import ArchitectAgent

# ==========================================
# CONFIGURATION
# ==========================================

# --- OPTION A: OpenAI API (Recommended for best results) ---
# PowerShell: $env:OPENAI_API_KEY = "sk-..."
# Linux/Mac:  export OPENAI_API_KEY="sk-..."
BASE_URL = None
API_KEY  = os.environ.get("OPENAI_API_KEY")
MODEL    = "gpt-4o-mini"   # or "gpt-4o" for stronger results

# --- OPTION B: Local LLM via llama.cpp (Free, Private, No API Key needed) ---
# 1. Download a GGUF model from: https://huggingface.co/models?search=gguf
# 2. Run: llama-server.exe -m your-model.gguf -c 2048 --port 8080
# 3. Uncomment the lines below and comment out OPTION A:
# BASE_URL = "http://localhost:8080/v1"
# API_KEY  = "sk-no-key"       # llama.cpp ignores this
# MODEL    = "local-model"      # llama.cpp ignores this
# Recommended: Qwen2.5-Coder-7B-Instruct-Q4, DeepSeek-Coder-6.7B

MAX_ITERATIONS = 50
MAX_EPOCHS_PER_RUN = 30    # Hard safety cap per architecture
EPOCH_PATIENCE = 3         # Stop training an arch if loss doesn't improve for this many epochs
# ==========================================
def main():
    os.makedirs('results', exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f'results/run_{run_id}.json'
    
    print("Loading Obscured Dataset...")
    train_loader, val_loader, n_features, n_classes = get_dataloaders()
    
    agent = ArchitectAgent(base_url=BASE_URL, api_key=API_KEY, model=MODEL)
    iteration_losses = []
    run_log = []  # Full persistent log
    
    print("\nStarting AITL Blind NAS Optimization Loop...")
    
    try:
        for i in range(1, MAX_ITERATIONS + 1):
            print(f"\n{'='*30}")
            print(f" ITERATION {i}")
            print(f"{'='*30}")
            
            print("> Agent is generating architecture...")
            try:
                code = agent.generate_model_code(n_features, n_classes, i)
            except Exception as e:
                print(f"> [!] API Error: {str(e).splitlines()[0]}")
                print(">     Retrying next iteration...")
                agent.add_feedback(i, val_loss=float('inf'), val_acc=0.0, error=str(e))
                iteration_losses.append(float('nan'))
                continue
            
            if code == "CONVERGED":
                print("\n[!] The AI Agent has analyzed the history and declared CONVERGENCE.")
                print("    Architectural limits reached. The LLM has autonomously stopped the AITL loop.")
                break
            
            print("> Executing training locally...")
            results, error = train_and_evaluate(code, train_loader, val_loader, 
                                                max_epochs=MAX_EPOCHS_PER_RUN,
                                                epoch_patience=EPOCH_PATIENCE)
            
            if error:
                print(f"> [!] Agent generated invalid code:\n      {error.splitlines()[-1]}")
                agent.add_feedback(i, val_loss=float('inf'), val_acc=0.0, error=error)
                iteration_losses.append(float('nan'))
                continue
                
            val_loss = results["final_val_loss"]
            val_acc = results["val_acc"]
            epochs_run = results["epochs_run"]
            print(f"> Results -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Epochs run: {epochs_run}")
            
            agent.add_feedback(i, val_loss, val_acc)
            is_best = agent.update_checkpoint(i, val_loss, code)
            if is_best:
                print(f"> [★] NEW BEST! Saving checkpoint at iteration {i}")
            else:
                print(f"> [STAGNATION: {agent.stagnation_counter}/5] No improvement since iter {agent.best_iteration}")
            iteration_losses.append(val_loss)
            
            # --- Persist to JSON after every iteration ---
            run_log.append({
                "iteration": i,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epochs_run": epochs_run,
                "is_best": is_best,
                "code": code,
            })
            with open(log_path, 'w') as f:
                json.dump({
                    "run_id": run_id,
                    "model": MODEL,
                    "best_loss": agent.best_loss,
                    "best_iteration": agent.best_iteration,
                    "best_code": agent.best_code,
                    "iterations": run_log
                }, f, indent=2)
            print(f"> [Log] Saved to {log_path}")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user — saving partial results...")
        
    print("\nAITL Optimization Complete. Plotting Results...")
    plot_results(iteration_losses)

def plot_results(losses, pivot_iterations=None):
    import numpy as np
    
    valid = [(i+1, l) for i, l in enumerate(losses) if l == l]  # filter NaN
    if not valid:
        print("No valid results to plot.")
        return
        
    xs = [x for x, _ in valid]
    ys = [y for _, y in valid]
    
    # Compute best-so-far envelope
    best_so_far = []
    cur_best = float('inf')
    for y in ys:
        cur_best = min(cur_best, y)
        best_so_far.append(cur_best)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    
    # Raw loss per iteration
    ax.plot(xs, ys, color='#58a6ff', linewidth=1.5, marker='o', markersize=4,
            alpha=0.7, label='Architecture Val Loss', zorder=2)
    
    # Best-so-far envelope
    ax.plot(xs, best_so_far, color='#3fb950', linewidth=2.5, linestyle='--',
            label='Best-So-Far (Improvement Frontier)', zorder=3)
    
    # Mark the global minimum
    min_loss = min(ys)
    min_iter = xs[ys.index(min_loss)]
    ax.scatter([min_iter], [min_loss], color='#f85149', s=120, zorder=5,
               label=f'Global Best: {min_loss:.4f} (iter {min_iter})')
    ax.annotate(f' Best: {min_loss:.4f}', xy=(min_iter, min_loss),
                xytext=(min_iter + 1, min_loss + 0.02),
                color='#f85149', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_title('AITL Blind NAS: Validation Loss over Agent Iterations',
                 color='white', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Successful Architectures Evaluated', color='#8b949e', fontsize=12)
    ax.set_ylabel('Validation Loss', color='#8b949e', fontsize=12)
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.8, linestyle='--')
    ax.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='white', fontsize=10)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/loss_curve.png', dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Plot saved to -> results/loss_curve.png  (Best loss: {min_loss:.4f} at iter {min_iter})")

if __name__ == "__main__":
    main()
