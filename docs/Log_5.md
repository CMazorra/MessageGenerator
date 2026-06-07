# Variant E: Advanced Heuristic Search (Lookahead, Semantic Guidance & Pruning)

## Experiment Configuration
- **Objective:** Address the shortcomings of the baseline Graph Search (Variant D), specifically the high latency, "myopic" length violations, and the loss of semantic tone during step-by-step generation.
- **Modification:** Upgraded the `_score_search_state` evaluation function to include three advanced AI planning techniques:
  1. **Length Lookahead (`_score_length_lookahead`):** A projection heuristic that penalizes branches early if they mathematically cannot satisfy `min_words` or `max_words` given their current depth and distance from a terminal state.
  2. **Continuous Semantic Guidance:** Replaced the end-of-branch binary judge with an intermediate lexical/vectorial tone scorer (`_score_semantic_guidance`). This calculates the gradient of semantic alignment at every step, keeping the tree search focused on the requested tone.
  3. **Dynamic Pruning (`_should_prune_state`):** Aggressively drops unpromising branches (e.g., branches that already contain forbidden words or have extremely low heuristic scores) to optimize the Beam Search width and drastically reduce inference latency.

## Comparative Results (Advanced Graph Search)

### Global Success and Latency
| Generator Model | Global Score | Strict Accuracy | Latency (Initial) | Retries Used |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3 (8B)** | 80.00% | 80.0% | 56.87s | 0.0 |
| **Mistral (7B)** | **95.00%** | **80.0%** | 56.83s | 0.0 |
| **Phi-3 (3.8B)** | 58.33% | 20.0% | **39.12s** | 0.0 |

### Detailed Sub-Metrics
| Metric | Llama 3 (8B) | Mistral (7B) | Phi-3 (3.8B) |
| :--- | :--- | :--- | :--- |
| **Lexical Compliance** | 80.0% | 90.0% | 70.0% |
| **Formatting Compliance** | 66.6% | **100.0%** | 66.6% |
| **Length Adherence** | **100.0%** | **100.0%** | 100.0% |
| **Semantic Success (Tone)**| **100.0%** | **100.0%** | 0.0% |

## Analysis of the Advanced Architecture

### 1. The Cure for Algorithmic Myopia (Length Lookahead)
The most striking improvement is Mistral's recovery in the **Length Adherence** metric. In Variant D, Mistral scored 50% because it blindly expanded branches until it hit the wall of the `max_words` limit. By introducing the Lookahead heuristic, the search algorithm now anticipates the constraint budget. Mistral's Length Adherence surged to a perfect **100%**, proving that lookahead projections successfully steer the LLM towards earlier, coherent terminal states.

### 2. Latency Optimization via Pruning
In Variant D, Mistral took an agonizing 75.18 seconds to complete its tree traversal. With the introduction of the `_should_prune_state` threshold, the algorithm now ruthlessly cuts off doomed branches (e.g., those that inadvertently generate a forbidden word at depth 1) without evaluating their subsequent children. This dynamic pruning slashed Mistral's latency down to **56.83 seconds**—a ~24% speed increase—while simultaneously raising its global score to an outstanding 95%. 

### 3. The Return of Empathy (Continuous Semantic Guidance)
In the baseline graph search, the piecemeal generation destroyed the emotional continuity of the text. By injecting `semantic_weight` into the intermediate node scoring, both Llama 3 and Mistral achieved a flawless **100% Semantic Success**. The algorithm now actively selects the "most empathetic" or "most dramatic" next sentence from the candidate pool, effectively fusing the rigid control of classical CSP algorithms with the stylistic flair of generative AI.

### 4. The Absolute Floor of Small Models (Phi-3)
Despite the sophisticated search architecture, Phi-3 remains stagnant at a 58.33% Global Score, 20% Strict Accuracy, and **0% Semantic Success**. The logs indicate that the 3.8B parameter model simply lacks the latent capacity to propose high-quality, diverse semantic candidate branches. Even when the algorithm perfectly scores and sorts the candidates, it is forced to choose from a pool of sub-par options. This confirms a fundamental principle for this architecture: **The heuristic search is only as good as the raw generation quality of the node-expander.** ## Final Conclusion for the Project
Variant E represents the pinnacle of this project's experimental design. We successfully transformed a generative AI prompt into a mathematically constrained Graph Search problem. By shifting the workload from "Prompt Engineering" to "Search Space Optimization" (using Lookahead, Pruning, and Vector Semantics), Mistral (7B) achieved a **95% Global Success Rate**—a near-perfect alignment of semantic creativity and strict algorithmic compliance. This architecture proves highly superior to both zero-shot prompting and single-pass reflection loops for complex constraint satisfaction in Natural Language Generation.