"""
AITL V2 — Autonomous ML Runner
Orchestrates the full AITL loop:
  1. Load real dataset (labels stripped)
  2. Agent generates ANY model code
  3. Trainer executes and measures
  4. Feed results back to agent
  5. Agent decides when to stop (no human iteration cap)

Supports both OpenAI API and local Ollama models.

Usage:
    # OpenAI (GPT-4o-mini)
    $env:OPENAI_API_KEY="your-key"
    python runner.py --backend openai --model gpt-4o-mini

    # Local (Qwen 2.5 Coder via Ollama)
    python runner.py --backend ollama --model qwen2.5-coder:7b
"""
import os
import sys
import json
import time
import argparse
import datetime
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import get_data
from trainer import execute_agent_code
from agent import AutonomousAgent


# ─── Safety cap (NOT a design parameter — just prevents infinite loops) ───
SAFETY_MAX_ITERATIONS = 200
MIN_ITERATIONS_BEFORE_STOP = 5
STAGNATION_PATIENCE = 5
TIMEOUT_PER_ITERATION = 120  # seconds


def run_experiment(backend="ollama", model_name="gemma4", api_key=None):
    """Run the full AITL V2 experiment."""
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 70)
    print("  AITL V2 — Autonomous ML Engineering Experiment")
    print(f"  Backend: {backend} | Model: {model_name}")
    print(f"  Started: {timestamp}")
    print("=" * 70)
    
    # ─── Step 1: Load data ───
    X_train, y_train, X_val, y_val, n_features, n_classes = get_data(
        n_samples=10000, seed=42
    )
    n_train, n_val = len(X_train), len(X_val)
    
    # ─── Step 2: Initialize agent ───
    if backend == "ollama":
        agent = AutonomousAgent(
            base_url="http://localhost:11434/v1",
            model=model_name
        )
        print(f"\n  [Config] Using LOCAL model: {model_name} via Ollama")
        print(f"  [Config] Cost: $0")
    elif backend == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            print("ERROR: Set OPENAI_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        agent = AutonomousAgent(api_key=key, model=model_name)
        print(f"\n  [Config] Using OPENAI model: {model_name}")
    else:
        print(f"ERROR: Unknown backend '{backend}'. Use 'ollama' or 'openai'.")
        sys.exit(1)
    
    # ─── Step 3: The AITL Loop ───
    all_results = []
    stop_reason = None
    start_time = time.time()
    
    print(f"\n  [Loop] Starting autonomous search (agent decides when to stop)")
    print(f"  [Loop] Safety cap: {SAFETY_MAX_ITERATIONS} iterations (failsafe only)")
    print(f"  [Loop] Min iterations: {MIN_ITERATIONS_BEFORE_STOP}")
    print("-" * 70)
    
    iteration = 0
    try:
        while iteration < SAFETY_MAX_ITERATIONS:
            iteration += 1
            iter_start = time.time()
            
            print(f"\n{'='*70}")
            print(f"  ITERATION {iteration}")
            print(f"{'='*70}")
            
            # Agent generates code
            try:
                code = agent.generate_code(
                    n_features=n_features,
                    n_classes=n_classes,
                    n_train=n_train,
                    n_val=n_val,
                    iteration=iteration,
                    min_iterations=MIN_ITERATIONS_BEFORE_STOP,
                    patience=STAGNATION_PATIENCE,
                    timeout=TIMEOUT_PER_ITERATION
                )
            except Exception as e:
                print(f"  [ERROR] LLM call failed: {e}")
                agent.add_feedback(iteration, 0, 0, "", "ERROR", error=str(e))
                continue
            
            # Check for STOP signal
            if code == "STOP":
                stop_reason = f"Agent autonomously stopped at iteration {iteration}"
                print(f"\n  [STOP] Agent decided to stop. Reason: converged.")
                print(f"  [STOP] Families explored: {agent.model_families_tried}")
                break
                
            # Hard cutoff if the model is too "optimistic" and won't stop itself
            if agent.stagnation_counter >= STAGNATION_PATIENCE * 3:
                stop_reason = f"System forced stop (Plateaued for {agent.stagnation_counter} iterations)"
                print(f"\n  [STOP] {stop_reason}. Agent failed to improve after 3 full pivot attempts.")
                break
            
            # Detect model family
            family = agent._detect_model_family(code)
            print(f"  [Model Family] {family}")
            
            # Execute the code
            print(f"  [Trainer] Executing agent code (timeout={TIMEOUT_PER_ITERATION}s)...")
            result, error = execute_agent_code(
                code, X_train, y_train, X_val, y_val, 
                n_classes, timeout=TIMEOUT_PER_ITERATION
            )
            
            iter_time = time.time() - iter_start
            
            if error:
                print(f"  [FAILED] {error[:200]}")
                agent.add_feedback(iteration, 0, 0, code, family, error=error)
                all_results.append({
                    "iteration": iteration,
                    "val_accuracy": None,
                    "val_loss": None,
                    "family": family,
                    "error": error[:500],
                    "time_seconds": round(iter_time, 1),
                    "code": code,
                })
                continue
            
            val_acc = result["val_accuracy"]
            val_loss = result["val_loss"]
            
            # Update checkpoint
            is_best = agent.update_checkpoint(iteration, val_loss, val_acc, code)
            agent.add_feedback(iteration, val_loss, val_acc, code, family)
            
            marker = " ★ NEW BEST ★" if is_best else ""
            print(f"  [RESULT] Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | "
                  f"Family: {family} | Time: {iter_time:.1f}s{marker}")
            
            if is_best:
                print(f"  [BEST] New best at iteration {iteration}!")
            else:
                print(f"  [BEST] Still: iter {agent.best_iteration} "
                      f"(acc={agent.best_acc:.4f}, stagnation={agent.stagnation_counter})")
            
            all_results.append({
                "iteration": iteration,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "family": family,
                "is_best": is_best,
                "time_seconds": round(iter_time, 1),
                "code": code,
            })
            
        else:
            stop_reason = f"Safety cap reached ({SAFETY_MAX_ITERATIONS} iterations)"
            
    except KeyboardInterrupt:
        stop_reason = f"Experiment manually aborted by user at iteration {iteration}"
        print(f"\n  [ABORT] Caught Ctrl+C. Saving partial results...")
    except Exception as e:
        stop_reason = f"Experiment crashed: {e}"
        print(f"\n  [ERROR] {traceback.format_exc()}")
    
    total_time = time.time() - start_time
    
    # ─── Step 4: Summary ───
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Stop reason: {stop_reason}")
    print(f"  Total iterations: {iteration}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best accuracy: {agent.best_acc:.4f}")
    print(f"  Best loss: {agent.best_loss:.4f}")
    print(f"  Best iteration: {agent.best_iteration}")
    print(f"  Model families explored: {agent.model_families_tried}")
    print(f"  Backend: {backend} ({model_name})")
    
    # ─── Step 5: Save results ───
    run_data = {
        "run_id": timestamp,
        "backend": backend,
        "model": model_name,
        "dataset": "covtype_10k_stripped",
        "n_features": n_features,
        "n_classes": n_classes,
        "n_train": n_train,
        "n_val": n_val,
        "best_accuracy": agent.best_acc,
        "best_loss": agent.best_loss,
        "best_iteration": agent.best_iteration,
        "best_code": agent.best_code,
        "total_iterations": iteration,
        "stop_reason": stop_reason,
        "total_time_seconds": round(total_time, 1),
        "model_families_explored": list(agent.model_families_tried),
        "iterations": all_results,
    }
    
    json_path = os.path.join(results_dir, f"v2_run_{backend}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(run_data, f, indent=2)
    print(f"\n  [Saved] Results: {json_path}")
    
    # ─── Step 6: Generate plot ───
    _generate_plot(all_results, agent, backend, model_name, results_dir, timestamp)
    
    return run_data


def _generate_plot(results, agent, backend, model_name, results_dir, timestamp):
    """Generate the AITL improvement frontier plot."""
    successful = [r for r in results if r["val_accuracy"] is not None]
    if not successful:
        print("  [Plot] No successful iterations to plot.")
        return
    
    iterations = [r["iteration"] for r in successful]
    accuracies = [r["val_accuracy"] for r in successful]
    families = [r["family"] for r in successful]
    
    # Best-so-far frontier
    best_so_far = []
    running_best = 0
    for acc in accuracies:
        running_best = max(running_best, acc)
        best_so_far.append(running_best)
    
    # Color map by model family
    family_colors = {
        'RandomForest': '#2ecc71',
        'GradientBoosting': '#e74c3c',
        'ExtraTrees': '#27ae60',
        'SVM': '#8e44ad',
        'LogisticRegression': '#f39c12',
        'KNN': '#1abc9c',
        'DecisionTree': '#d35400',
        'sklearn_MLP': '#2980b9',
        'PyTorch_NN': '#e67e22',
        'AdaBoost': '#c0392b',
        'Ensemble': '#16a085',
        'Unknown': '#95a5a6',
    }
    colors = [family_colors.get(f, '#95a5a6') for f in families]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Scatter by family
    ax.scatter(iterations, accuracies, c=colors, s=80, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)
    
    # Best-so-far line
    ax.plot(iterations, best_so_far, 'g--', linewidth=2, alpha=0.8, label='Best-So-Far', zorder=2)
    
    # Mark global best
    if agent.best_iteration:
        ax.scatter([agent.best_iteration], [agent.best_acc], 
                   c='red', s=200, marker='*', zorder=5, 
                   label=f'Best: {agent.best_acc:.4f} (iter {agent.best_iteration})')
    
    # Legend for families
    unique_families = list(set(families))
    for fam in sorted(unique_families):
        color = family_colors.get(fam, '#95a5a6')
        ax.scatter([], [], c=color, s=60, label=fam)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(f'AITL V2 — Autonomous ML Engineering\n{backend}/{model_name} | '
                 f'Agent-controlled stopping', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(results_dir, f"v2_plot_{backend}_{timestamp}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AITL V2 — Autonomous ML Engineering")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model", default="gemma4",
                        help="Model name (default: gemma4 for Ollama, gpt-4o-mini for OpenAI)")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Dataset subsample size (default: 10000)")
    
    args = parser.parse_args()
    
    # Set default model based on backend
    if args.backend == "openai" and args.model == "gemma4":
        args.model = "gpt-4o-mini"
    
    run_experiment(
        backend=args.backend,
        model_name=args.model,
        api_key=args.api_key,
    )
