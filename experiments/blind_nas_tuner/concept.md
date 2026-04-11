# Concept: The "Blind" NAS Tuner

## The Problem with LLM Evaluation

If we want to prove that an **AI In The Loop (AITL)** system can autonomously improve through a continuous feedback loop, we have to rule out "cheating."

If we ask an LLM (like GPT-4 or Claude) to design a perfect neural network for the MNIST dataset, it doesn't need a feedback loop. It has memorized the optimal architecture for MNIST during its massive pre-training phase. It can output a highly accurate ConvNet zero-shot. This proves the LLM's parametric knowledge, **not** the validity of an AITL feedback loop.

## The Solution: "Blind" Neural Architecture Search (NAS)

To mathematically prove that AITL works via empirical feedback (and not pre-training bias), we must blind the LLM to the dataset. 

1. **Obfuscation**: We provide the LLM with an obscured dataset ("Dataset X"). We only give it the input tensor shape (e.g., `(1, 28, 28)` for vision or `(128)` for tabular) and the output classification space (e.g., `Num Classes: 10`). 
2. **Tabula Rasa**: The LLM does not know if it's looking at handwriting, medical imagery, or sensor data. It cannot rely on its pre-existing intuition for standard datasets.

## The AITL Loop in Action

The Blind NAS Tuner operates as an autonomous multi-step loop over $N$ iterations:

1. **Self-Generating**: The LLM agent generates a PyTorch model architecture and hyperparameter configuration (Learning Rate, Optimizer, Epochs) as a `JSON` configuration or python string.
2. **Execution**: A local python runner instantiates the generated PyTorch `nn.Module` and trains it on Dataset X for a rapid micro-training session (e.g., 5 epochs).
3. **Feedback Mechanism**: The runner calculates the validation loss and accuracy curve over those 5 epochs and returns this data to the LLM agent.
4. **Self-Improving**: The LLM agent ingests the validation results, analyzes the loss curve (e.g., "The loss is plateauing too quickly, I should add BatchNorm and increase the LR"), and generates a **new** architecture for the next iteration.

## Why This Proves AITL

By removing the LLM's ability to zero-shot the answer, the **only** way the validation loss can trend downwards over 20 iterations is if the system is successfully utilizing the empirical feedback loop to learn and adapt.

If we plot the best validation loss achieved at each iteration, we expect to see a downward trend. This curve acts as the mathematical and irrefutable proof of the AITL **Self-Improving** property, allowing us to publish our core thesis without revealing proprietary, domain-specific systems like ASRT.
