import re
from openai import OpenAI

SYSTEM_PROMPT = """You are an AI Architecture Search Agent (AITL Pattern).
Your task is to design a PyTorch neural network to minimize validation loss on an obscured dataset.
You DO NOT know what the dataset is. You only know:
- Input features shape: ({n_features})
- Output classes: {n_classes}

Available imports already provided: `torch`, `nn` (torch.nn), `F` (torch.nn.functional), `optim` (torch.optim).
Do NOT add `import` statements. They are already available.

You have two options:
OPTION 1: Output a Python code block containing exactly:
- `class Model(nn.Module):`
- `def get_optimizer(model):`
Output ONLY the code inside ```python ... ```. No explanations.

OPTION 2 (ONLY available after {min_iters}+ iterations): If the loss has fully converged and no further improvement is possible, output exactly: CONVERGED
"""

PIVOT_PROMPT = """You are an AI Architecture Search Agent (AITL Pattern).
Your task is to design a PyTorch neural network to minimize validation loss on an obscured dataset.
Input: {n_features} features. Output: {n_classes} classes.
Available: `torch`, `nn`, `F`, `optim`. Do NOT add import statements.

*** STRATEGY RESET TRIGGERED ***
Your last {patience} attempts ALL failed to improve on your best result.

Your BEST architecture so far (from iteration {best_iter}, Val Loss: {best_loss:.4f}):
```python
{best_code}
```

Approaches you already TRIED that did NOT work (last {patience} iterations):
{failed_approaches}

You MUST try a COMPLETELY DIFFERENT strategy. Do NOT repeat what failed.
Think about:
- Completely different network topology (e.g. residual connections, wider/narrower layers)
- Different regularization (BatchNorm, Dropout, L2 weight decay)
- Different optimizer (SGD+momentum, AdamW, RMSprop with different LR)
- Different activation functions (LeakyReLU, ELU, GELU, Tanh)
- Ensemble-like architectures (multiple parallel branches)

Output ONLY the new code inside ```python ... ```. No explanations.
"""

class ArchitectAgent:
    def __init__(self, base_url=None, api_key=None, model="gpt-4o"):
        self.is_local = base_url is not None
        if self.is_local:
            self.client = OpenAI(base_url=base_url, api_key=api_key or "sk-no-key-required", timeout=300.0)
        else:
            self.client = OpenAI(api_key=api_key, timeout=300.0)
            
        self.model = model
        self.history = []
        
        # Checkpoint tracking
        self.best_loss = float('inf')
        self.best_code = None
        self.best_iteration = None
        self.stagnation_counter = 0
        self.failed_approaches_since_best = []

    def update_checkpoint(self, iteration, val_loss, code):
        """Call this when a successful iteration completes."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_code = code
            self.best_iteration = iteration
            self.stagnation_counter = 0
            self.failed_approaches_since_best = []
            return True  # New best!
        else:
            self.stagnation_counter += 1
            # Record what was tried (first 3 lines as summary)
            summary = "\n".join(code.splitlines()[:5]) + "..."
            self.failed_approaches_since_best.append(f"  - Iter {iteration}: {summary}")
            return False

    def generate_model_code(self, n_features, n_classes, iteration, min_iterations=5, patience=5):
        """Main entry point. Auto-pivots strategy if plateau is detected."""
        
        # PLATEAU DETECTED: pivot to a completely different strategy
        if self.stagnation_counter >= patience and self.best_code is not None:
            print(f"\n  [PIVOT] Plateau detected ({patience} iters without improvement).")
            print(f"  [PIVOT] Resetting to best (iter {self.best_iteration}, loss {self.best_loss:.4f}) with new strategy.")
            return self._generate_pivot_code(n_features, n_classes, patience)
        
        # Normal iteration: explore/improve
        return self._generate_normal_code(n_features, n_classes, iteration, min_iterations)

    def _generate_normal_code(self, n_features, n_classes, iteration, min_iterations):
        prompt = f"Iteration: {iteration}\n\n"
        if self.history:
            prompt += "History of previous architectures (last 10):\n"
            for h in self.history[-10:]:
                prompt += f"  Iter {h['iteration']}: "
                if h['error']:
                    prompt += f"FAILED -> {h['error'][:80]}\n"
                else:
                    marker = " *** BEST ***" if h['val_loss'] == self.best_loss else ""
                    prompt += f"Val Loss: {h['val_loss']:.4f} | Val Acc: {h['val_acc']:.4f}{marker}\n"
            
            prompt += f"\nCurrent best: Val Loss {self.best_loss:.4f} at iteration {self.best_iteration}.\n"
            prompt += "Analyze the trend and try to beat the best loss. Adjust architecture/hyperparameters accordingly."
            if self.history[-1].get('error'):
                prompt += "\nCRITICAL: The last model FAILED. Fix the PyTorch bug before trying new things."
        else:
            prompt += "First iteration. Generate a simple but solid baseline model."
            
        system_str = SYSTEM_PROMPT.format(n_features=n_features, n_classes=n_classes, min_iters=min_iterations)
        if iteration < min_iterations:
            system_str = system_str.split("OPTION 2")[0].strip()
            system_str += f"\n\nYou MUST output code. Minimum {min_iterations} iterations required. Keep exploring."
        
        return self._call_llm(system_str, prompt)

    def _generate_pivot_code(self, n_features, n_classes, patience):
        failed_str = "\n".join(self.failed_approaches_since_best) if self.failed_approaches_since_best else "  (see history above)"
        
        system_str = PIVOT_PROMPT.format(
            n_features=n_features,
            n_classes=n_classes,
            patience=patience,
            best_iter=self.best_iteration,
            best_loss=self.best_loss,
            best_code=self.best_code,
            failed_approaches=failed_str
        )
        prompt = f"Generate a COMPLETELY DIFFERENT architecture. Do not repeat any of the failed approaches."
        
        # Reset stagnation counter so we explore again after the pivot
        self.stagnation_counter = 0
        self.failed_approaches_since_best = []
        
        return self._call_llm(system_str, prompt)

    def _call_llm(self, system_str, prompt):
        try:
            if self.is_local:
                full_prompt = f"<|im_start|>system\n{system_str}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                response = self.client.completions.create(
                    model=self.model, prompt=full_prompt,
                    temperature=0.7, max_tokens=1024, stop=["<|im_end|>", "```\n\n"]
                )
                raw_output = response.choices[0].text
            else:
                messages = [
                    {"role": "system", "content": system_str},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=0.7, max_tokens=1024
                )
                raw_output = response.choices[0].message.content
                
            print(f"  [DEBUG] Raw LLM output ({len(raw_output)} chars):\n{raw_output[:300]}...")
        except Exception as e:
            print("\n[API ERROR DETAILS]")
            if hasattr(e, 'response'):
                print("Status Code:", getattr(e.response, 'status_code', 'Unknown'))
                try: print("Response Body:", getattr(e.response, 'text', 'None'))
                except: pass
            print("===================\n")
            raise e
            
        return self._extract_code(raw_output)
        
    def add_feedback(self, iteration, val_loss, val_acc, error=None):
        self.history.append({
            "iteration": iteration,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "error": error
        })
        
    def _extract_code(self, text):
        if "CONVERGED" in text.strip().upper() and len(text.strip()) < 20:
            return "CONVERGED"
        
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        code = text.strip()
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"```\s*$", "", code)
        return code.strip()
