# Variant F: Monte Carlo Tree Search (MCTS) with Generative Rollouts

## Experiment Configuration
- **Objective:** Integrate discrete-event and stochastic simulation concepts (MCTS) into the NLP generation pipeline to solve the "Semantic Disconnect" observed in previous graph search variants. 
- **Modification:** 1. Replaced the deterministic Beam Search with a **Monte Carlo Tree Search (MCTS)** algorithm guided by the UCT formula ($UCT = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N_i}{n_i}}$).
  2. **Rollout Policy Upgrade:** Replaced the rigid, deterministic "word-padding" rollout with an `llm_rollout_complete` function. During the simulation phase, a lightweight LLM prompt naturally completes the partial sentence to the end of the budget.
  3. **Continuous Semantic Reward:** Adjusted the reward function to heavily weight the semantic embedding score evaluated on the natural rollout text, rather than a fragmented state.

## Comparative Results (MCTS Architecture)

### Global Success and Latency
| Generator Model | Global Score | Strict Accuracy | Latency (Initial) | Retries Used |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3 (8B)** | 88.33% | 60.0% | 507.38s | 0.0 |
| **Mistral (7B)** | **90.00%** | **60.0%** | 401.99s | 0.0 |
| **Phi-3 (3.8B)** | 88.33% | **60.0%** | **425.78s** | 0.0 |

### Detailed Sub-Metrics
| Metric | Llama 3 (8B) | Mistral (7B) | Phi-3 (3.8B) |
| :--- | :--- | :--- | :--- |
| **Lexical Compliance** | **100.0%** | 90.0% | **100.0%** |
| **Formatting Compliance** | 66.6% | **100.0%** | **100.0%** |
| **Length Adherence** | **100.0%** | 75.0% | **100.0%** |
| **Semantic Success (Tone)**| 66.6% | **100.0%** | 33.3% |

## Analysis of the MCTS Architecture

### 1. The Latency Catastrophe (O(n) LLM Calls)
The most glaring result of this experiment is the total collapse of system performance. Latency skyrocketed from an average of ~50 seconds (Variant E) to between **400 and 500 seconds per instance**. 
In standard game theory (like Hex or Chess agents), rollouts are lightning-fast mathematical simulations. In this architecture, executing a generative LLM rollout for *every expanded node* turns the simulation phase into an enormous bottleneck. Generating text is auto-regressive; forcing the system to simulate multiple possible futures using a neural network is computationally unviable without heavy parallelization or a much faster baseline rollout policy (e.g., Markov Chains).

### 2. Semantic Recovery (The Rollout Effect)
Despite the latency, the hypothesis regarding semantic poisoning was proven correct. By allowing the LLM to organically finish the sentence during the MCTS rollout, the embedding evaluator finally had realistic text to judge. 
- **Mistral** jumped from a failing semantic score to a perfect **100% Semantic Success**, proving that it can successfully balance constraints if the reward signal is accurate.
- The UCT selection successfully guided the root node to pick the most "empathetic" or "dramatic" starting branches based on the simulated futures.

### 3. The Trade-off: Semantic Focus vs. Length Adherence
While Mistral achieved perfect formatting and semantics, its **Length Adherence dropped to 75%** (down from 100% in Variant E). 
This is a classic MCTS reward balancing issue. Because the algorithm now highly values the *Semantic Reward* of a natural-sounding rollout, it occasionally favors branches that have an excellent tone but slightly miss the strict `max_words` limit. The continuous semantic gradient overpowered the binary hard-constraint penalty in the backpropagation phase.

### 4. Strict Accuracy Ceiling
Across all three models, Strict Accuracy remains hard-capped at **60.0%**. While individual constraints are being met at high rates (90-100%), achieving all of them simultaneously in a single tree path remains evasive. When the algorithm fixes formatting, it loses tone; when it optimizes tone via MCTS, it loses length adherence.

## Conclusion and Next Steps
MCTS is a theoretically beautiful approach to Natural Language Generation, solving the intermediate semantic evaluation problem through stochastic rollouts. However, the 8-minute latency renders it useless for any practical software application. 
To finalize the project, the system must either:
1. Revert to the deterministic Heuristic Search (Variant E) while implementing a faster, non-generative approach to Semantic Guidance.
2. Implement an ultra-fast statistical model (e.g., n-grams or a tiny quantized model) exclusively for the MCTS rollouts to bring latency back under 60 seconds.