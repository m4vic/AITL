# Blind NAS Tuner

This directory contains the Proof-of-Concept (PoC) for the Artificial Intelligence In The Loop (AITL) paper. 

**Goal**: Prove the AITL self-improving property mathematically by forcing an LLM to optimize a neural network via empirical feedback loops, negating its ability to guess the answer via pre-training bias.

Please read [`concept.md`](concept.md) first to understand the methodology and necessity of this architecture.

## Structure

*   `concept.md`: Detailed explanation of the blinding methodology.
*   `agent.py`: (Coming soon) The LLM agent logic that generates PyTorch configurations.
*   `trainer.py`: (Coming soon) The local runner that evaluates the generated PyTorch models.
*   `runner.py`: (Coming soon) The orchestrator linking the agent and the trainer.
