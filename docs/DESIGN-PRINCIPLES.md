# Design Principles for AITL Systems

Based on successful AITL systems like AlphaZero, Constitutional AI, and Auto-Research, we extract the following reusable design principles.

## 1. Frozen Baseline + Adaptive System
Maintain one component that never changes (a frozen baseline) while allowing others to adapt. This enables mathematical comparison over time. Without a frozen baseline, it is impossible to detect regression or model drift.

## 2. Confidence-Thresholded Feedback
Only feed high-confidence results back into the training or optimization loop. Flag low-confidence outputs for human review. This prevents "feedback poisoning."

## 3. Multi-Perspective Evaluation
Use ensembles of evaluators, rather than a single judge. Combining different models or logic-flows reduces single-points-of-failure in AI judgment.

## 4. Measurable Convergence Criteria
Define explicit, mathematically comparable metrics that indicate success or failure. The AITL agent itself should be capable of reading this metric and determining when the feedback loop has converged to prevent oscillation.

## 5. Hierarchical Human Oversight
Humans should audit aggregates, delta tracking, and outliers, not every instance. For example, humans monitor an aggregate "loss curve" or "safety delta," rather than reading the individual generated data points.

## 6. Adversarial Testing of the AITL System Itself
Red-team your own evaluation system. AITL systems develop blind spots; adversarial testing ensures the AI judges aren't being trivially fooled by formatting or simple tricks.
