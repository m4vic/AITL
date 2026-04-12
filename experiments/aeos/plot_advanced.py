import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Use an academic style if seaborn is available, else standard clean matplotlib
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
except ImportError:
    pass

OLLAMA_JSON = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aitl_v2\results\v2_run_ollama_20260412_132700.json"
OPENAI_JSON = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aitl_v2\results\v2_run_openai_20260412_131308.json"

# Output path to results directory
OUTPUT_DIR = r"f:\AI-IN-THE-LOOP\aitl-paper\experiments\aitl_v2\results"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "comparative_frontier.png")

def extract_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    iterations = []
    actual_accuracies = []
    best_accuracies = []
    running_best = 0.0
    
    for row in data['iterations']:
        it = row['iteration']
        acc = row.get('val_accuracy')
        if acc is not None:
            if acc > running_best:
                running_best = acc
            iterations.append(it)
            actual_accuracies.append(acc)
            best_accuracies.append(running_best)
            
    return iterations, actual_accuracies, best_accuracies, data['model']

def plot_comparison():
    plt.figure(figsize=(12, 7.5), dpi=300)
    
    # Extract data
    iter_ollama, actual_ollama, best_ollama, model_ollama = extract_data(OLLAMA_JSON)
    iter_openai, actual_openai, best_openai, model_openai = extract_data(OPENAI_JSON)
    
    # BANANA COLOR PALETTE 🍌
    # Local (Qwen): Fresh Banana Yellow (Deep Gold so it shows on white)
    BANANA_YELLOW = '#D4AC0D'
    # Cloud (GPT): Bruised Banana Brown
    BANANA_BROWN = '#5C3A21'
    
    # 1. Plot Actual fluctuating attempts (scatter/faded line)
    plt.plot(iter_openai, actual_openai, 'o-', color=BANANA_BROWN, alpha=0.25, linewidth=1.5, markersize=5, label=f'Cloud Attempts ({model_openai})')
    plt.plot(iter_ollama, actual_ollama, 's-', color=BANANA_YELLOW, alpha=0.4, linewidth=1.5, markersize=5, label=f'Local Attempts ({model_ollama})')

    # 2. Plot stepped lines for "Best so far Frontier" (Solid bold)
    plt.step(iter_openai, best_openai, where='post', label=f'Cloud Frontier', 
             color=BANANA_BROWN, linewidth=3.5, zorder=5)
             
    plt.step(iter_ollama, best_ollama, where='post', label=f'Local Frontier', 
             color=BANANA_YELLOW, linewidth=3.5, zorder=6)
             
    # 3. Add Area Shading under the frontiers for an ultra-premium look
    plt.fill_between(iter_openai, best_openai, 0.7, step='post', color=BANANA_BROWN, alpha=0.08)
    plt.fill_between(iter_ollama, best_ollama, 0.7, step='post', color=BANANA_YELLOW, alpha=0.15)
             
    # Annotate end points with matching banana colors
    plt.annotate(f'Aborted\nIter {iter_openai[-1]}', 
                 (iter_openai[-1], best_openai[-1]),
                 textcoords="offset points", xytext=(-15,-20), ha='center', fontsize=10, fontweight='bold', color=BANANA_BROWN)
                 
    plt.annotate(f'Auto-Stopped\nIter {iter_ollama[-1]}', 
                 (iter_ollama[-1], best_ollama[-1]),
                 textcoords="offset points", xytext=(-15,15), ha='center', fontsize=10, fontweight='bold', color='#B7950B')

    # Formatting for academic & premium look
    plt.title('Autonomous ML Experiment: Search Behavior & Improvement Frontier', fontsize=16, fontweight='bold', pad=20, color='#333333')
    plt.xlabel('Experiment Iteration', fontsize=13, fontweight='bold', color='#444444')
    plt.ylabel('Validation Accuracy', fontsize=13, fontweight='bold', color='#444444')
    
    # Clean grid
    plt.grid(True, linestyle='--', alpha=0.6, color='#BDC3C7')
    
    # Legend Styling
    plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True, facecolor='#FDFEFE', edgecolor='#EEEEEE')
    
    # Set limits to make shading look grounded
    plt.xlim(1, max(iter_openai[-1], iter_ollama[-1]) + 2)
    plt.ylim(0.71, 0.815) # Framed slightly above minimum attempts
    
    # Remove top and right spines for a modern look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Plot saved to {OUTPUT_IMAGE}")

if __name__ == '__main__':
    plot_comparison()
