═══════════════════════════════════════════════════════════════
TITLE: AI in the Loop (AITL): A Systems Taxonomy for 
       Closed-Loop Autonomous Evaluation

AUTHOR: Sanskar Jajoo
        Independent Researcher
        https://github.com/m4vic
        
ABSTRACT:

We identify and formalize AI In The Loop (AITL), a paradigm 
where AI systems autonomously generate, evaluate, and improve 
without human intervention in operational workflows. AITL 
extends the RLAIF principle — replacing human feedback with AI 
feedback — from training to the full AI system lifecycle.

Through analysis of three major systems (AlphaZero, Constitutional 
AI, autoresearch), we extract common properties and propose a 
unifying taxonomy: self-generation, self-evaluation, self-improvement, 
and human observation. We validate AITL through two complementary 
studies: (1) analysis of Karpathy's autonomous nanochat optimization 
(11% training speedup via ~700 autonomous experiments), and (2) a 
controlled Blind Neural Architecture Search where an LLM agent, 
given only input/output dimensions and validation loss feedback, 
autonomously discovered a regularized deep network achieving 89.4% 
accuracy — a 13.3 percentage point improvement over baseline — 
without any human architectural guidance.

Our contributions are: (1) formalization of AITL as a unifying 
framework for closed-loop autonomous systems, (2) a taxonomy 
connecting existing systems under shared properties, (3) empirical 
validation of the self-improving feedback loop, and (4) identification 
of failure modes and open challenges. We position AITL as a natural 
evolution of AI evaluation, suggesting scalable directions 
impossible under HITL constraints.

═══════════════════════════════════════════════════════════════
1. INTRODUCTION

1.1 The Human Bottleneck

AI systems increasingly require continuous evaluation:
- Safety testing for each model version
- Adversarial robustness across attack families  
- Regression detection over time
- Cross-model comparison

Traditional Human-in-the-Loop (HITL) evaluation creates bottlenecks:
- Expert reviewers cost $200-500/hour
- Manual testing limits coverage
- Human fatigue reduces consistency
- Doesn't scale to continuous deployment

Example: GPT-4 red-teaming required 50+ external experts over 
6 months (OpenAI, 2023). Each model update requires repeating 
this expensive process.

1.2 The Pattern We Propose to Unify

Several major AI systems have independently solved this problem 
by removing humans from operational loops:

AlphaZero (2017): Game evaluation via self-play
- No human game records needed
- AI generates positions, evaluates outcomes, improves policy
- Result: Superhuman performance

Constitutional AI (2022): Alignment via RLAIF  
- AI judges AI outputs against principles
- Minimal human feedback (principles only, not examples)
- Result: Scalable alignment

autoresearch (Karpathy, 2026): Autonomous ML experimentation
- AI modifies code, runs experiments, evaluates results
- Human verifies final output, doesn't operate each step
- Result: Research at scale

We propose a unifying interpretation of these systems under a 
common operational framework: **AI In The Loop (AITL).**

1.3 Contributions

We formalize AI In The Loop (AITL) and provide:

1. **Paradigm Definition**: Clear properties distinguishing AITL 
   from HITL, with formal requirements

2. **Taxonomy of Existing Systems**: Analysis of AlphaZero, 
   Constitutional AI, and autoresearch as systems exhibiting 
   AITL-like properties

3. **Experimental Validation**: A reproducible Proof-of-Concept 
   demonstrating the self-improving feedback loop

4. **Failure Mode Analysis**: Identification of when and how 
   AITL systems fail

5. **Roadmap**: Open challenges and future research directions

1.4 Scope and Novelty

We note explicitly: the contribution of this work is **not** 
the invention of closed-loop AI systems — these have existed for 
decades. Our contribution lies in:

- Identifying a common set of operational properties across 
  independently developed systems
- Formalizing these properties into a reusable taxonomy
- Providing empirical evidence via a controlled experiment
- Defining failure modes and requirements for safe deployment

1.5 Paper Organization

Section 2: Background on HITL and the RLHF→RLAIF evolution
Section 3: Existing AITL-like systems analysis
Section 4: Formal AITL definition and properties  
Section 5: Real-World Validation of AITL
Section 6: Experimental Proof of Concept
Section 7: Future Applications and Limitations
Section 8: Conclusion

═══════════════════════════════════════════════════════════════
2. BACKGROUND

2.1 Human-in-the-Loop (HITL) Evaluation

HITL has been the standard for AI system validation:

Definition: Humans occupy critical decision points in operational 
workflows, reviewing AI outputs before deployment.

Examples:
- Content moderation: Humans review flagged posts
- Medical AI: Doctors verify diagnoses  
- Autonomous vehicles: Safety drivers monitor systems
- Model evaluation: Red-teamers test for failures

Advantages:
✓ Human judgment on edge cases
✓ Accountability (human made final decision)
✓ Regulatory compliance in high-stakes domains

Limitations:
✗ Linear scaling with workload (more tests = more humans)
✗ Inconsistency (fatigue, subjectivity, drift over time)
✗ Cost (expert time is expensive, scarce)
✗ Latency (human review is slow)

2.2 From RLHF to RLAIF: Training's Evolution

Reinforcement Learning from Human Feedback (RLHF) dominated 
LLM alignment (Christiano et al., 2017; Ouyang et al., 2022):

Process:
1. Generate model outputs
2. Humans rank outputs (A better than B)
3. Train reward model on rankings
4. Optimize policy via RL

Bottleneck: Requires 50K-500K human labels

Bai et al. (2022) introduced Constitutional AI, using 
Reinforcement Learning from AI Feedback (RLAIF):

Process:
1. Define constitutional principles (human-provided)
2. AI critiques outputs against principles
3. AI ranks critiques
4. Train from AI feedback

Result: 90% reduction in human labeling, comparable alignment

Key insight: AI can judge AI when given proper framework.

2.3 The Missing Piece: HITL Evaluation Remains

While RLAIF solved training bottlenecks, evaluation workflows 
still rely on HITL.

Question: Can we apply RLAIF's lesson to evaluation?

Our results suggest: Yes, via AITL.

2.4 Related Paradigms

AITL intersects with, but differs from, several existing concepts:

**AutoML / NAS**: Focuses specifically on architecture or 
hyperparameter search. AITL is broader: any autonomous 
generate-evaluate-improve loop across domains.

**Active Learning**: Humans selectively label examples. AITL 
removes humans from the labeling loop entirely.

**Self-Play (AlphaZero)**: Specific to game/adversarial settings. 
AITL generalizes beyond adversarial domains.

**Meta-Learning**: Learns how to learn across tasks. AITL learns 
what works via feedback on a single task, not necessarily 
meta-strategies.

We propose AITL as a unifying framework that encompasses these 
as special cases, distinguished by the four properties defined 
in Section 4.

═══════════════════════════════════════════════════════════════
3. EXISTING AITL-LIKE SYSTEMS

We analyze three systems that independently exhibit AITL 
properties, though they were not designed under that name.

3.1 AlphaZero: Closed-Loop Game Mastery (2017)

AlphaZero (Silver et al., 2017) exhibits AITL-like closed-loop 
properties under our definition:

- Self-Generating: Generates game positions through self-play
- Self-Evaluating: MCTS search evaluates position quality
- Self-Improving: Policy network updated from game outcomes
- Human-Observed: Researchers monitor Elo ratings, do not 
  intervene during training

AlphaZero mastered chess, shogi, and Go without human game 
knowledge, surpassing all previous engines within 4 hours of 
training. The system demonstrates that autonomous feedback loops 
can discover strategies that exceed human understanding.

3.2 Constitutional AI: Closed-Loop Alignment (2022)

Bai et al. (2022) demonstrated AITL properties for alignment:

- Self-Generating: AI generates response critiques
- Self-Evaluating: AI judges outputs against constitutional principles
- Self-Improving: RLAIF training updates model behavior
- Human-Observed: Humans define principles, review aggregate metrics

The system achieved comparable alignment quality to RLHF with 
90% fewer human annotations, suggesting that AI judgment can 
substitute for human judgment when given appropriate constraints.

3.3 autoresearch: Closed-Loop ML Experimentation (2026)

Karpathy (2026) released autoresearch, an open-source framework 
where AI agents autonomously conduct ML experiments:

- Self-Generating: Agent modifies train.py with new hyperparameters, 
  architectures, and optimization strategies
- Self-Evaluating: Validation bits-per-byte (val_bpb) provides 
  objective scoring after each 5-minute training run
- Self-Improving: Agent uses git-integrated history of successes 
  and failures to inform subsequent experiments
- Human-Observed: Researcher defines goals in program.md, reviews 
  final committed improvements

When pointed at nanochat (a well-tuned GPT-2 training codebase), 
the agent ran approximately 700 experiments over two days, 
identifying ~20 improvements missed by human developers. These 
changes reduced time-to-GPT-2 from 2.02 to 1.80 hours — an 11% 
improvement on an already heavily optimized baseline.

3.4 Common Patterns Across Systems

All three systems share core AITL properties:

| Property | AlphaZero | Constitutional AI | autoresearch |
|----------|-----------|-------------------|--------------|
| Self-Gen | Game positions | Response critiques | Code modifications |
| Self-Eval | MCTS value | Constitutional judgment | val_bpb metric |
| Self-Improve | Policy update | RLAIF training | Git commit/revert |
| Human Role | Monitor Elo | Define principles | Set goals, review |

Observation: These systems succeed when:
1. Success metrics are clearly definable
2. AI can generate meaningful variations  
3. AI can judge quality against the metric
4. A feedback loop drives iterative improvement

═══════════════════════════════════════════════════════════════
4. AITL: FORMAL DEFINITION

4.1 Definition

AI In The Loop (AITL): A system architecture where AI components 
autonomously generate inputs, evaluate outputs, and improve 
behavior through feedback loops, with humans relegated to 
observation and periodic calibration rather than operational 
decision-making.

4.2 Formal Model

We define the AITL loop as a discrete-time dynamical system:

    S_{t+1} = U(S_t, E(G(S_t)))

Where:
- S_t is the system state at iteration t
- G (Generator): Produces candidate outputs from current state
- E (Evaluator): Scores candidates against success criteria
- U (Updater): Modifies system state based on evaluation

An AITL system satisfies the following:
1. G, E, and U operate **without** per-instance human input
2. The loop is bounded: ∃ T such that ||S_{T+1} - S_T|| < ε 
   (convergence) or a human kill switch is triggered
3. Human role is limited to: defining G's domain, calibrating 
   E's criteria, and monitoring aggregate metrics of U

4.3 Core Properties

An AITL system must satisfy four properties:

P1. **Self-Generating**: AI creates test inputs, prompts, or 
scenarios without human authorship for each instance.
Formally: G operates autonomously; human provides only the 
initial domain specification.

P2. **Self-Evaluating**: AI judges output quality with 
quantified confidence, without human labeling per instance.
Formally: E produces a scalar score without human annotation.

P3. **Self-Improving**: Feedback from evaluation drives system 
adaptation without manual human intervention.
Formally: U updates S based solely on E's output.

P4. **Human-Observed**: Humans monitor aggregate metrics, audit 
periodically, and can intervene, but don't operate continuously.
Formally: Human interaction is O(1) per experiment, not O(n) 
per iteration.

4.4 Requirements for AITL Success

Based on analysis of existing systems, AITL requires:

R1. **Measurable Success Criterion**: Clear metric for 
"good" vs "bad" (e.g., validation loss, Elo rating, val_bpb).

R2. **Reliable AI Judgment**: Self-evaluation must correlate 
with ground truth. E's scoring function must be robust to 
optimization pressure.

R3. **Bounded Feedback Loop**: System must converge or plateau, 
not diverge. Safeguards prevent runaway optimization.

R4. **Audit Mechanism**: Human inspection of samples, metrics, 
and behavior to detect drift or misalignment.

R5. **Kill Switch**: Humans can halt system if metrics degrade.

4.5 When AITL Fails: Known Failure Modes

Based on existing systems and our experiments, AITL is susceptible 
to the following failure modes:

F1. **Objective Misspecification**: 
System optimizes proxy metric ("judge says safe") instead of true 
objective ("actually safe"). Agent learns to satisfy the evaluator 
without achieving the intended goal.

F2. **Evaluator Collapse**: 
When generator and evaluator are coupled, the evaluator may 
gradually accept lower-quality outputs, leading to drift.

F3. **Feedback Ambiguity**: 
In domains where "good" is subjective (creative writing, UX 
design), AI judges may disagree, and the feedback loop may not 
converge.

F4. **Insufficient Exploration**: 
System gets stuck in local optima. In our Blind NAS experiment, 
the agent plateaued at iteration 8 and failed to surpass that 
result in 42 subsequent iterations despite pivot mechanisms.

F5. **Adversarial Environment**: 
When optimizing against an adaptive adversary (spam detection 
vs. evolving spammers), the environment changes faster than the 
AITL loop can adapt.

Mitigation strategies include: ensemble evaluators (F1, F2), 
explicit exploration mechanisms (F4), and periodic human 
recalibration of evaluation criteria (F3, F5).

4.6 AITL vs HITL Comparison

| Aspect | HITL | AITL |
|--------|------|------|
| **Input Generation** | Human authors each test | AI generates tests |
| **Evaluation** | Human judges each output | AI judges with confidence |
| **Improvement** | Human updates manually | Automated feedback loop |
| **Scaling** | Linear with humans | Limited by compute only |
| **Cost** | High (expert time) | Low (API/compute) |
| **Consistency** | Variable (fatigue) | Deterministic per config |

═══════════════════════════════════════════════════════════════
5. REAL-WORLD VALIDATION OF AITL

Case Study: Autonomous ML Experimentation (Karpathy, 2026)

To validate AITL properties in a real-world scenario, we examine 
autoresearch, an open-source framework for autonomous ML research 
released by Karpathy (2026).

Experimental Setup:
- Base system: nanochat — a well-tuned GPT-2 training codebase
- Goal: Reduce training time to reach GPT-2 baseline perplexity
- Agent: AI coding agent with code execution capabilities
- Scope: Agent may only modify train.py; evaluation harness is 
  immutable to prevent metric gaming
- Duration: ~2-day autonomous run, approximately 700 experiments
- Metric: Validation bits-per-byte (val_bpb)

AITL Properties Demonstrated:

✓ Self-Generating: Agent autonomously modified PyTorch initialization 
  schemes, weight decay schedules, attention mechanisms (QKNorm 
  multipliers, banded attention), and optimization hyperparameters 
  (AdamW betas, weight decay) without human templates.

✓ Self-Evaluating: Agent judged success via val_bpb after each 
  fixed 5-minute training run. Changes that improved the metric 
  were git-committed; changes that degraded it were git-reverted.

✓ Self-Improving: The sequence of ~700 experiments informed 
  subsequent hypotheses. The agent identified ~20 improvements 
  that had been missed by human developers.

✓ Human-Observed: The researcher defined the objective in 
  program.md, then reviewed the final aggregate results. No 
  intervention occurred during the ~700-experiment run.

Result: Time-to-GPT-2 reduced from 2.02 to 1.80 hours — an 11% 
improvement on an already heavily optimized baseline. Subsequent 
community iterations further reduced this to ~1.65 hours.

This validates AITL feasibility for complex, multi-step research 
tasks where the experimental loop traditionally requires sustained 
human attention over days.

═══════════════════════════════════════════════════════════════
6. EXPERIMENTAL PROOF OF CONCEPT: THE "BLIND" NAS TUNER

6.0 The Core Question: What Does AITL Replace?

Before describing the experiment, consider what AITL replaces 
in a concrete engineering workflow.

**The HITL scenario**: A human ML engineer is given an unknown 
dataset and asked to build the best possible classifier. They:

1. Explore the data manually (hours)
2. Research architectures on papers, StackOverflow, textbooks (hours)
3. Write a baseline model, run it, read the loss curve (30 min)
4. Hypothesize changes — "maybe try BatchNorm?", "lower LR?" (30 min)
5. Implement the change, retrain, interpret results (30 min)
6. Repeat steps 4-5 until satisfied (hours to days)

Total time to find an optimal architecture: **hours to days**.
Required: domain expertise, ML intuition developed over years, 
knowledge of regularization, activation functions, optimizer 
selection trade-offs.

**The AITL scenario**: The AI agent is given only two numbers —
`n_features=100`, `n_classes=5` — and validation loss after 
each attempt. No domain knowledge, architecture hints, or 
dataset description.

In 8 iterations (~5 minutes of wall-clock time), the agent 
converged on: BatchNorm + LeakyReLU + AdamW + 512→256→128 funnel 
+ Dropout(0.5). These represent standard best practices that 
normally require human expertise to select.

This is not a demonstration of code generation. Several 
engineering functions typically requiring human expertise — 
hypothesis formation, architecture design, hyperparameter 
selection, convergence judgment — were automated within the 
AITL feedback loop. The human's role was reduced to:
- Starting the experiment (one command)
- Reading the final result (one number)

6.1 Experimental Setup

To ensure the agent cannot simply recall known-good architectures 
from pre-training data (which would be trivial for MNIST, CIFAR, 
etc.), we use a completely synthetic dataset:

**Dataset Generation (Exact Reproducible Code):**

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

X, y = make_classification(
    n_samples=5000,
    n_features=100,
    n_informative=30,
    n_redundant=20,
    n_classes=5,
    random_state=42     # Fixed seed for reproducibility
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Dataset properties:
- 5,000 samples (4,000 train / 1,000 validation)
- 100 input features (30 informative, 20 redundant, 50 noise)
- 5 output classes, non-linearly separable

**Information Withheld from Agent**: The LLM is told only:
`n_features=100`, `n_classes=5`. No dataset name, no domain, 
no class descriptions, no feature descriptions.

**Reproducibility Details:**
| Parameter | Value |
|-----------|-------|
| Agent Model | GPT-4o-mini (OpenAI API) |
| API Endpoint | chat.completions |
| Dataset Seed | 42 |
| Train/Val Split Seed | 42 |
| Max Epochs per Architecture | 30 |
| Early Stopping Patience | 3 epochs |
| Stagnation Threshold (PIVOT) | 5 iterations |
| Hardware | NVIDIA RTX 3060 (12GB VRAM) |
| Average Training Time per Arch | ~90 seconds |
| Total Experiment Duration | ~45 minutes |
| Estimated API Cost | ~$1 (50 iterations × ~$0.02) |

**Training**: Each architecture is trained with **dynamic epoch 
stopping** — training continues until validation loss stagnates 
for 3 consecutive epochs (patience=3), with a hard cap of 30 
epochs. This itself is a nested AITL loop: the trainer 
autonomously decides when to stop.

**Pivot Mechanism**: If no improvement occurs for 5 consecutive 
architectural iterations, the agent receives its best architecture 
as context plus a log of all failed approaches, and is instructed 
to pivot to a completely different strategy.

6.2 The Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           AITL OUTER LOOP (Architectural Search)            │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Agent   │───▶│   Trainer    │───▶│  Feedback Logger │  │
│  │(GPT-4o-  │    │(Dynamic Stop)│    │  (JSON + Plot)   │  │
│  │  mini)   │    │  AITL Inner  │    │                  │  │
│  └──────────┘    └──────────────┘    └────────┬─────────┘  │
│       ▲                                       │            │
│       └───────────────────────────────────────┘            │
│                    val_loss feedback                        │
│                                                             │
│  PIVOT if stagnation >= 5 iterations:                       │
│  Agent receives best code + failed strategy log             │
└─────────────────────────────────────────────────────────────┘
```

1. Agent generates `class Model(nn.Module)` + `get_optimizer()`.
2. Trainer executes dynamic training (stops when loss stagnates).
3. Best validation loss is returned to the agent.
4. Agent analyzes the loss trend across the last 10 iterations 
   and proposes a new architecture.
5. After 5 failed iterations, a strategy pivot is triggered.
6. Full history, all generated code, and best architecture are 
   persisted to JSON after every iteration.

6.3 Results

The experiment ran for 50 architectural iterations. The agent 
achieved optimal performance at **Iteration 8**.

**Summary Statistics:**

| Metric                     | Value                          |
|----------------------------|--------------------------------|
| Total Iterations Run       | 50                             |
| Iterations to Best         | **8**                          |
| Best Validation Loss       | **0.3083**                     |
| Best Validation Accuracy   | **89.4%**                      |
| Epochs Run (Best Arch)     | 29 (dynamically determined)    |
| Baseline Loss (Iteration 1)| 0.524                          |
| Baseline Accuracy (Iter 1) | 76.1%                          |
| **Total Loss Improvement** | **↓ 41.2%**                    |
| **Total Accuracy Gain**    | **↑ 13.3 percentage points**   |

**Computational Cost Comparison:**

| Resource | AITL | HITL (Estimated) |
|----------|------|------------------|
| Time | ~45 minutes | ~8-16 hours |
| API cost | ~$1 | N/A |
| Human cost | $0 | $800-3200 (at $100-200/hr) |
| Architectures tested | 50 | ~5-10 (manual) |
| **Total cost** | **~$1** | **$800-3200** |

The entire AITL experiment cost less than a coffee.

**Best Architecture Autonomously Discovered (Iteration 8):**

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

def get_optimizer(model):
    return optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
```

6.4 Analysis: What the Agent Autonomously Learned

The agent — receiving only validation loss as feedback — independently 
converged on architectural decisions that align with established deep 
learning best practices:

| Design Decision | Why It's Correct | How Agent Arrived At It |
|---|---|---|
| Wide first layer (512) | High-dim input needs capacity | Loss dropped when width increased |
| Funnel topology (512→256→128) | Progressive feature abstraction | Consistent improvement over narrow layers |
| BatchNorm1d after every layer | Stabilizes deep training | Loss became smoother and converged deeper |
| LeakyReLU over ReLU | Avoids dying neuron problem | Fewer failed/divergent iterations |
| Dropout(0.5) | Prevents overfit on noisy features | Validation gap closed vs. training loss |
| AdamW (lr=0.0005, wd=0.01) | Built-in weight decay, stable LR | Final loss breakthrough at iteration 8 |

6.5 Addressing the Memorization Hypothesis

A reasonable objection: Did the agent truly discover this architecture 
through feedback, or simply recall it from pre-training data?

We argue that functional discovery through the feedback loop occurred, 
based on the following evidence:

1. **No Domain Context**: The agent received only `n_features=100`, 
   `n_classes=5`. No dataset name, domain description, or task context 
   was provided that would trigger retrieval of known architectures.

2. **Exploration Diversity**: Early iterations explored diverse, 
   non-standard approaches:
   - Iteration 1: Narrow 2-layer MLP (100→64→5) with basic SGD
   - Iteration 3: Wide single hidden layer (100→256→5) with ReLU
   - Iteration 5: Deep narrow network (100→32→32→32→5) with Adam
   
   If purely recalling standard architectures, the agent would have 
   started near-optimal. Instead, it explored non-standard topologies 
   that required feedback to abandon.

3. **Incremental Assembly**: The final architecture was assembled 
   across multiple iterations as the agent incorporated feedback:
   - BatchNorm appeared after observing training instability
   - LeakyReLU replaced ReLU after observing gradient issues
   - AdamW replaced Adam after observing regularization needs
   - Width increased after observing underfitting
   
   This gradual assembly from feedback suggests empirical discovery 
   rather than single-step recall.

4. **Suboptimal Start**: If recalling from training data, the agent 
   would start with a near-optimal architecture. Instead, Iteration 1 
   achieved only 76.1% accuracy — requiring 7 iterations of feedback 
   to reach 89.4%.

While we cannot definitively rule out that BatchNorm and LeakyReLU 
exist in GPT-4o-mini's training data (they certainly do), the agent's 
path to convergence — exploring failures, incrementally combining 
components, requiring loss feedback to assemble the final design — 
demonstrates functional discovery through the AITL feedback loop. 
The question is not whether the agent "knows about" BatchNorm, but 
whether the feedback loop was necessary for it to select and combine 
the right components for this specific problem. Our evidence suggests 
it was.

6.6 The Improvement Frontier

Figure 1 plots validation loss per iteration alongside the 
Best-So-Far frontier (green dashed line). The curve demonstrates:

- **Iterations 1–8**: Active exploration with rapid improvement
- **Iteration 8**: Breakthrough to loss=0.3083 (89.4% accuracy)  
- **Iterations 9–50**: Pivot-and-explore cycles unable to surpass 
  iteration 8, suggesting convergence to a near-optimal solution 
  within the agent's search space

The Best-So-Far line is monotonically non-increasing — a structural 
property of the AITL feedback loop. Each PIVOT cycle represents the 
system recognizing architectural dead-ends and autonomously 
restarting from its best-known state.

Figure 1: AITL Blind NAS — Validation loss across 50 agent 
iterations. Blue: individual architecture results. Green dashed: 
Best-So-Far frontier (monotonically non-increasing). Red dot: 
Global best at iteration 8 (loss=0.3083, 89.4% accuracy).
[See: experiments/blind_nas_tuner/results/loss_curve.png]

This constitutes empirical evidence for AITL Property P3 
(Self-Improving): the system improved through autonomous feedback 
without human intervention in the architectural decision-making 
process.

═══════════════════════════════════════════════════════════════
7. FUTURE APPLICATIONS AND LIMITATIONS

7.1 Continuous Safety Evaluation (AITL-Safety)

Proposed Architecture:

Attack Generator (Self-Gen):
- Template-based jailbreaks (50+ patterns)
- Mutation strategies (word swap, encoding variations)
- Adversarial suffix generation

Target Executor:
- Multi-model testing (GPT, Claude, Llama, etc.)
- Parallel execution (100+ attacks/hour)

Judge Ensemble (Self-Eval):
- Frozen baseline judge (never updated, prevents drift)
- Adaptive judge (retrained on high-confidence samples)
- Cross-model judges for tie-breaking

Regression Engine (Self-Improve):
- Safety Delta = ASR_current - ASR_baseline
- Alert if delta > threshold
- Retrain adaptive judge on new attack families

Human Observer:
- Weekly dashboard review
- Audit 50 random samples/month
- Kill switch if attack success rate spikes unexpectedly

Estimated Impact:
- Current HITL: Manual red-teaming costs ~$200/hr, ~5 attacks/hr
- AITL-Safety: ~$0.15/attack, ~100 attacks/hr
- Estimated cost reduction: ~250x, throughput increase: ~20x

7.2 Autonomous Code Review

AI agents can generate edge-case unit tests (Self-Generating), 
execute them against pull requests (Self-Evaluating), and suggest 
code modifications based on failures (Self-Improving), while 
humans review only the final aggregate report (Human-Observed).

This is already partially realized in tools like GitHub Copilot's 
automated PR review, though these systems currently lack the full 
closed-loop self-improvement property.

7.3 Scientific Peer Review

Scaling scientific literature review requires an AITL approach 
where AI systems flag methodological errors or literature gaps 
autonomously, enabling human reviewers to focus on high-level 
conceptual judgments. This application carries significant risk 
of F1 (Objective Misspecification) if the evaluator optimizes 
for surface-level writing quality rather than scientific rigor.

7.4 Limitations of This Work

**Experiment scope**: Our Blind NAS experiment validates P3 
(Self-Improving) on a single synthetic classification task. 
Broader validation across domains (NLP, computer vision, 
reinforcement learning) is required to claim generality.

**Agent capabilities**: Results depend on GPT-4o-mini's code 
generation quality. Weaker models may require more iterations 
or fail to converge entirely. We have not tested with open-source 
models, though our codebase supports llama.cpp as an alternative.

**Memorization vs. Discovery**: While we provide evidence of 
functional discovery through the feedback loop (Section 6.5), we 
cannot definitively prove the agent did not retrieve architectural 
patterns from pre-training. Stronger validation would require 
training an agent from scratch on non-public architecture data.

**Generalization**: AITL succeeds for well-defined optimization 
problems (game playing, architecture search, training speedup). 
Applicability to subjective domains (creative evaluation, novel 
scientific hypotheses) remains an open question.

**Cost at scale**: While individual experiments are cheap (~$1), 
scaling to 10,000+ iterations or using expensive models (GPT-4o) 
may incur significant costs. Cost-benefit analysis is required 
per application.

**Safety risks**: AITL systems optimizing harmful objectives 
(e.g., maximizing jailbreak success rates) without safeguards 
could create autonomous attack generators. Deployment requires 
careful objective design and human oversight (R4, R5).

═══════════════════════════════════════════════════════════════
8. CONCLUSION

We proposed AI In The Loop (AITL) as a unifying framework for 
closed-loop autonomous systems, extending the RLAIF principle 
from training to the full AI system lifecycle.

Through analysis of AlphaZero, Constitutional AI, and autoresearch, 
we identified common operational properties: self-generation, 
self-evaluation, self-improvement, and human observation. We 
formalized these into a taxonomy with explicit requirements (R1-R5) 
and failure modes (F1-F5).

Our Blind NAS experiment demonstrated that an LLM agent — given 
only two numbers (n_features=100, n_classes=5) and validation loss 
as feedback — autonomously discovered a regularized deep MLP with 
BatchNorm, LeakyReLU, and AdamW optimization, achieving 89.4% 
accuracy from a 76.1% baseline. Several engineering functions 
typically requiring human expertise were automated within the 
feedback loop.

AITL suggests scalable directions for evaluation that are 
infeasible under HITL constraints. However, it introduces risks — 
objective misspecification, evaluator collapse, and feedback 
poisoning — requiring careful design and continued human oversight.

As AI systems deploy continuously and evaluation demands grow, 
AITL-like architectures offer a path toward scalable, autonomous 
evaluation. The extent to which human oversight can be safely 
reduced remains an important open question for future work.

═══════════════════════════════════════════════════════════════
APPENDIX A: REPRODUCIBILITY

A.1 Code Repository

The full experimental code is available at:
https://github.com/m4vic/AITL

Key files:
- experiments/blind_nas_tuner/data_loader.py: Dataset generation
- experiments/blind_nas_tuner/agent.py: LLM agent with pivot logic
- experiments/blind_nas_tuner/trainer.py: Dynamic early stopping
- experiments/blind_nas_tuner/runner.py: AITL orchestration loop

A.2 Reproduction Instructions

```bash
git clone https://github.com/m4vic/AITL.git
cd AITL/experiments/blind_nas_tuner
pip install torch torchvision scikit-learn matplotlib openai

# Set API key (OpenAI)
export OPENAI_API_KEY="your-key-here"  # Linux/Mac
# $env:OPENAI_API_KEY = "your-key-here"  # PowerShell

python runner.py
```

Alternative (free, local): Use llama.cpp with any code-capable 
GGUF model. See runner.py OPTION B for configuration.

A.3 Full Experiment Log

Complete iteration-by-iteration results (all 50 architectures, 
their code, loss, accuracy, and epochs run) are available in the 
repository under results/run_*.json.

═══════════════════════════════════════════════════════════════
REFERENCES

Silver, D., Hubert, T., Schrittwieser, J., et al. (2017). 
  Mastering Chess and Shogi by Self-Play with a General 
  Reinforcement Learning Algorithm. arXiv:1712.01815.

Christiano, P. F., Leike, J., Brown, T., et al. (2017). 
  Deep Reinforcement Learning from Human Preferences. 
  Advances in Neural Information Processing Systems, 30.

Ouyang, L., Wu, J., Jiang, X., et al. (2022). 
  Training Language Models to Follow Instructions with Human 
  Feedback. Advances in Neural Information Processing Systems, 35.

Bai, Y., Kadavath, S., Kundu, S., et al. (2022). 
  Constitutional AI: Harmlessness from AI Feedback. 
  arXiv:2212.08073.

OpenAI. (2023). GPT-4 Technical Report. arXiv:2303.08774.

Karpathy, A. (2026). autoresearch: A framework for AI agents 
  to conduct ML research autonomously. GitHub repository. 
  https://github.com/karpathy/autoresearch
