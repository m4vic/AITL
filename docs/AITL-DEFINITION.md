# Formal Definition of AI In The Loop (AITL)

## Definition
**AI In The Loop (AITL)** is a system architecture where AI components autonomously generate inputs, evaluate outputs, and improve behavior through continuous feedback loops. In an AITL architecture, humans are relegated to observation and periodic calibration rather than operational decision-making.

## Core Properties

An AITL system must satisfy four core properties:

### P1. Self-Generating
The AI creates test inputs, prompts, code configs, or scenarios autonomously without human authorship for each instance.
- *Anti-Pattern (HITL)*: A human writes every single unit test or red-team attack prompt.

### P2. Self-Evaluating
The AI judges output quality with quantified metrics or modeled confidence, without human labeling per instance.
- *Anti-Pattern (HITL)*: Humans rank outputs manually (RLHF) or click "approve/reject" on logs.

### P3. Self-Improving
Feedback from the evaluation phase drives system adaptation without manual human intervention. The AI analyzes historical trends and mathematically adjusts the system.
- *Anti-Pattern (HITL)*: A human reviews evaluation reports and manually pushes an update to codebase.

### P4. Human-Observed
Humans monitor aggregate metrics, perform periodic audits, and control stopping criteria (kill switches), but they do not operate the pipeline continuously.
- *Anti-Pattern (HITL)*: A human acts as a critical choke-point in the deployment pipeline.

## Requirements for Success
1. **Measurable Success Criterion**: A mathematically definable metric for "good" vs "bad" (e.g., Validation Loss, Elo Rating, Attack Success Rate).
2. **Reliable AI Judgment**: The AI evaluator must correlate highly with ground-truth human judgment on the specific domain.
3. **Bounded Feedback Loop**: The system must have mechanisms to prevent runaway divergence and recognize convergence.
