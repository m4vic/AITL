"""
AITL V2 — Autonomous ML Agent
Uses any LLM (OpenAI API or Ollama local) to autonomously:
  1. Choose model families (sklearn, PyTorch, etc.)
  2. Design architectures and hyperparameters
  3. Decide when to stop (no human-set iteration cap)
"""
import re
from openai import OpenAI

SYSTEM_PROMPT = """You are an Autonomous ML Engineering Agent (AITL Pattern).

You have a classification dataset. Here is everything you know:
- n_features = {n_features}
- n_classes = {n_classes}
- Training samples: {n_train}
- Validation samples: {n_val}
- Features are numbered [0, 1, 2, ..., {max_feature}]. You do NOT know what they represent.
- Classes are numbered [0, 1, 2, ..., {max_class}]. You do NOT know what they represent.

YOUR TASK: Write a Python function `solve(X_train, y_train, X_val, y_val)` that:
1. Builds and trains ANY model you choose
2. Returns predictions as a numpy array of shape (n_val,) with integer class labels

You have full freedom to use ANY approach:
- sklearn: RandomForestClassifier, GradientBoostingClassifier, SVC, LogisticRegression, KNeighborsClassifier, ExtraTreesClassifier, MLPClassifier, etc.
- PyTorch: nn.Module, custom neural networks
- numpy: raw implementations
- Any combination or ensemble

IMPORTANT: The data is RAW (not preprocessed). You decide whether to:
- Standardize/normalize features
- Apply PCA or other dimensionality reduction
- Handle feature distributions
- Any other preprocessing you think is necessary

Available in your namespace: numpy (as np), sklearn submodules, torch, nn, F, optim.
You CAN import additional sklearn submodules if needed.

RULES:
1. You MUST define: def solve(X_train, y_train, X_val, y_val): ... return predictions
2. predictions must be a numpy array of integers in range [0, {max_class}]
3. Your code has a {timeout}-second time limit. Be efficient.
4. Output ONLY the code inside ```python ... ```. No explanations.

{stop_instruction}"""

STOP_INSTRUCTION_ACTIVE = """STOPPING OPTION:
If you have thoroughly explored multiple approaches and believe no further improvement 
is likely, output EXACTLY the word: STOP
Think like a pragmatic ML Engineer. Do NOT waste compute budget chasing 0.001% gains. 
Consider stopping if:
- You have tried at least 3 different model families (e.g. Trees, Deep Learning, SVM)
- Your recent pivot attempts completely failed to beat your best result
- You realize your baseline (e.g. Iteration 1 or 2) was actually optimal"""

STOP_INSTRUCTION_LOCKED = """You MUST output code. Minimum {min_iters} iterations required before stopping is allowed. Keep exploring."""

PIVOT_PROMPT = """You are an Autonomous ML Engineering Agent. Dataset: {n_features} features, {n_classes} classes, {n_train} training samples.

*** STRATEGY PIVOT TRIGGERED ***
Your last {patience} attempts ALL failed to beat your best result. Do not suffer from sunk-cost fallacy.

YOUR BEST RESULT (iteration {best_iter}, Accuracy: {best_acc:.4f}, Loss: {best_loss:.4f}):
```python
{best_code}
```

APPROACHES THAT FAILED (last {patience}):
{failed_approaches}

You MUST try something COMPLETELY DIFFERENT. Ideas:
- If you've only tried neural nets → try sklearn models (RandomForest, GradientBoosting, SVC)
- If you've only tried tree models → try neural networks or ensembles
- If you've tried both → try ensembles, different preprocessing, or exotic models
- Try: stacking, bagging, feature engineering, PCA + model, etc.

Output ONLY new code inside ```python ... ```. No explanations.

{stop_instruction}"""


class AutonomousAgent:
    def __init__(self, base_url=None, api_key=None, model="gpt-4o-mini"):
        """
        Initialize agent with LLM backend.
        
        For OpenAI: AutonomousAgent(api_key="sk-...", model="gpt-4o-mini")
        For Ollama: AutonomousAgent(base_url="http://localhost:11434/v1", model="gemma4")
        """
        self.is_local = base_url is not None
        if self.is_local:
            self.client = OpenAI(
                base_url=base_url, 
                api_key=api_key or "ollama",
                timeout=300.0
            )
        else:
            self.client = OpenAI(api_key=api_key, timeout=300.0)
            
        self.model = model
        self.history = []
        
        # Autonomous tracking
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_code = None
        self.best_iteration = None
        self.stagnation_counter = 0
        self.failed_approaches_since_best = []
        self.model_families_tried = set()  # Track what families were explored

    def update_checkpoint(self, iteration, val_loss, val_acc, code):
        """Update best-known result. Returns True if new best."""
        # Track model family
        family = self._detect_model_family(code)
        if family:
            self.model_families_tried.add(family)
        
        if val_acc > self.best_acc or (val_acc == self.best_acc and val_loss < self.best_loss):
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.best_code = code
            self.best_iteration = iteration
            self.stagnation_counter = 0
            self.failed_approaches_since_best = []
            return True
        else:
            self.stagnation_counter += 1
            summary_lines = code.strip().splitlines()[:6]
            summary = "\n".join(summary_lines) + "..."
            self.failed_approaches_since_best.append(
                f"  Iter {iteration} (acc={val_acc:.4f}): {summary}"
            )
            return False

    def _detect_model_family(self, code):
        """Detect which model family the agent used."""
        code_lower = code.lower()
        if 'randomforest' in code_lower:
            return 'RandomForest'
        elif 'gradientboosting' in code_lower:
            return 'GradientBoosting'
        elif 'extratrees' in code_lower:
            return 'ExtraTrees'
        elif 'svc' in code_lower or 'svm' in code_lower:
            return 'SVM'
        elif 'logisticregression' in code_lower:
            return 'LogisticRegression'
        elif 'kneighbors' in code_lower:
            return 'KNN'
        elif 'decisiontree' in code_lower:
            return 'DecisionTree'
        elif 'mlpclassifier' in code_lower:
            return 'sklearn_MLP'
        elif 'nn.module' in code_lower or 'nn.linear' in code_lower:
            return 'PyTorch_NN'
        elif 'adaboost' in code_lower:
            return 'AdaBoost'
        elif 'votingclassifier' in code_lower or 'stackingclassifier' in code_lower:
            return 'Ensemble'
        return 'Unknown'

    def generate_code(self, n_features, n_classes, n_train, n_val, 
                      iteration, min_iterations=5, patience=5, timeout=120):
        """Main entry point. Auto-decides: normal, pivot, or stop."""
        
        # Check for pivot
        if self.stagnation_counter >= patience and self.best_code is not None:
            print(f"\n  [PIVOT] Plateau detected ({patience} iters without improvement).")
            print(f"  [PIVOT] Best: iter {self.best_iteration}, acc={self.best_acc:.4f}")
            print(f"  [PIVOT] Families tried so far: {self.model_families_tried}")
            return self._generate_pivot_code(n_features, n_classes, n_train, n_val,
                                              iteration, min_iterations, patience, timeout)
        
        # Normal iteration
        return self._generate_normal_code(n_features, n_classes, n_train, n_val,
                                           iteration, min_iterations, timeout)

    def _generate_normal_code(self, n_features, n_classes, n_train, n_val,
                               iteration, min_iterations, timeout):
        """Generate code for a normal iteration."""
        
        # Build stop instruction based on iteration count
        if iteration >= min_iterations:
            stop_instr = STOP_INSTRUCTION_ACTIVE
        else:
            stop_instr = STOP_INSTRUCTION_LOCKED.format(min_iters=min_iterations)
        
        system_str = SYSTEM_PROMPT.format(
            n_features=n_features, n_classes=n_classes,
            n_train=n_train, n_val=n_val,
            max_feature=n_features - 1, max_class=n_classes - 1,
            timeout=timeout, stop_instruction=stop_instr
        )
        
        # Build user prompt with history
        prompt = f"Iteration: {iteration}\n\n"
        
        if self.history:
            prompt += "History of previous attempts (most recent last):\n"
            for h in self.history[-15:]:
                prompt += f"  Iter {h['iteration']}: "
                if h.get('error'):
                    prompt += f"FAILED -> {h['error'][:100]}\n"
                else:
                    marker = " *** BEST ***" if h.get('is_best') else ""
                    family = h.get('family', '?')
                    prompt += f"Acc: {h['val_acc']:.4f} | Loss: {h['val_loss']:.4f} | Model: {family}{marker}\n"
            
            prompt += f"\nCurrent best: Acc {self.best_acc:.4f} (iter {self.best_iteration})"
            prompt += f"\nFamilies explored: {', '.join(self.model_families_tried) or 'none yet'}"
            prompt += "\n\nAnalyze the trend. Try to beat the best accuracy."
            
            if self.history[-1].get('error'):
                prompt += "\nWARNING: Your last code FAILED. Fix the bug before trying new strategies."
        else:
            prompt += "First iteration. Start with a reasonable baseline model."
        
        return self._call_llm(system_str, prompt)

    def _generate_pivot_code(self, n_features, n_classes, n_train, n_val,
                              iteration, min_iterations, patience, timeout):
        """Generate code after a plateau is detected."""
        failed_str = "\n".join(self.failed_approaches_since_best[-patience:]) \
            if self.failed_approaches_since_best else "  (see history)"
        
        if iteration >= min_iterations:
            stop_instr = STOP_INSTRUCTION_ACTIVE
        else:
            stop_instr = STOP_INSTRUCTION_LOCKED.format(min_iters=min_iterations)
        
        system_str = PIVOT_PROMPT.format(
            n_features=n_features, n_classes=n_classes,
            n_train=n_train, patience=patience,
            best_iter=self.best_iteration, best_acc=self.best_acc,
            best_loss=self.best_loss, best_code=self.best_code,
            failed_approaches=failed_str, stop_instruction=stop_instr
        )
        prompt = "Generate a COMPLETELY DIFFERENT approach. Do NOT repeat what failed."
        
        # Reset stagnation for next cycle
        self.stagnation_counter = 0
        self.failed_approaches_since_best = []
        
        return self._call_llm(system_str, prompt)

    def _call_llm(self, system_str, prompt):
        """Call the LLM (OpenAI or Ollama) and return generated code."""
        try:
            messages = [
                {"role": "system", "content": system_str},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            raw_output = response.choices[0].message.content
            
            print(f"  [LLM] Response ({len(raw_output)} chars):")
            # Print first 400 chars for debugging
            for line in raw_output[:400].splitlines():
                print(f"    {line}")
            if len(raw_output) > 400:
                print(f"    ... ({len(raw_output) - 400} more chars)")
                
        except Exception as e:
            print(f"\n  [LLM ERROR] {type(e).__name__}: {str(e)[:200]}")
            raise e
        
        return self._extract_code(raw_output)

    def add_feedback(self, iteration, val_loss, val_acc, code, family, error=None):
        """Record iteration result for history."""
        self.history.append({
            "iteration": iteration,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "family": family,
            "error": error,
            "is_best": (val_acc == self.best_acc and val_loss == self.best_loss) if not error else False,
        })

    def _extract_code(self, text):
        """Extract Python code or STOP signal from LLM output."""
        stripped = text.strip()
        
        # Check for STOP signal
        if stripped.upper() == "STOP" or stripped.upper().startswith("STOP"):
            # Make sure it's actually a stop and not code starting with "STOP"
            if len(stripped) < 50 and 'def ' not in stripped:
                return "STOP"
        
        # Extract code from markdown code block
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Fallback: clean up raw text
        code = stripped
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"```\s*$", "", code)
        return code.strip()
