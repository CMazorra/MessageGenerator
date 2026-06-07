# Variant D: Heuristic Graph Search Generation (Basic Implementation)

## Experiment Configuration
- **Objective:** Shift the paradigm from post-generation correction (Reflection Loop) to a proactive constraint satisfaction planning model. Formulate Natural Language Generation (NLG) as a Graph Search problem where constraints are evaluated at each step to prevent rule-breaking before it occurs.
- **Modification:** Replaced the iterative full-text rewriting loop with a step-by-step expansion algorithm (`generate_with_search`). The LLM is now used strictly as a node generator (proposing $k$ next branches/sentences). The Python backend evaluates the `hard_score` of each partial state and selects the best path.

## Comparative Results (Graph Search Baseline)

### Global Success and Latency
| Generator Model | Global Score | Strict Accuracy | Latency (Initial) | Retries Used |
| :--- | :--- | :--- | :--- | :--- |
| **Llama 3 (8B)** | 75.00% | 60.0% | 44.21s | 0.0 |
| **Mistral (7B)** | **83.33%** | **60.0%** | 75.18s | 0.0 |
| **Phi-3 (3.8B)** | 58.33% | 20.0% | **29.55s** | 0.0 |

### Detailed Sub-Metrics
| Metric | Llama 3 (8B) | Mistral (7B) | Phi-3 (3.8B) |
| :--- | :--- | :--- | :--- |
| **Lexical Compliance** | 80.0% | 90.0% | 70.0% |
| **Formatting Compliance** | 66.6% | **100.0%** | 66.6% |
| **Length Adherence** | **100.0%** | 50.0% | **100.0%** |
| **Semantic Success (Tone)**| 66.6% | 66.6% | 0.0% |

## Analysis of the Graph Search Architecture

### 1. The Latency Explosion
By transitioning to a step-by-step graph search, we effectively eliminated the need for complete text retries (`retries_used = 0.0`). However, the trade-off is a massive increase in base latency. Mistral, which provides the highest global score, requires over 75 seconds to traverse the search tree. This happens because the model is invoked sequentially to generate and expand multiple branches at every depth level.

### 2. The "Myopic" Search Problem (Mistral's Length Failure)
Mistral demonstrated incredible syntactic control, achieving a perfect **100% in Formatting Compliance** and **90% in Lexical Compliance**. The Python-guided search successfully forced the model to close JSON brackets and include mandatory words. 
However, Mistral failed dramatically at **Length Adherence (50%)**. Because our current heuristic only evaluates the *present* state of the node, the algorithm is "myopic". It does not penalize a branch for approaching the `max_words` limit until it has already crossed it, leading the search path into dead-ends where it cannot organically finish the sentence without violating the length constraint.

### 3. The Semantic Disconnect (Phi-3's Tone Failure)
Phi-3 completely failed the subjective tone evaluation (**0% Semantic Success**). In the previous Variant C, the model generated the text in a single cohesive pass, allowing it to maintain an emotional or dramatic narrative. In this Graph Search variant, text is built piecemeal. Because we are only evaluating *Hard Constraints* at each intermediate step (deferring the Soft Constraint 'tone' evaluation until the very end), the smaller models lose the global semantic "vibe" of the prompt, resulting in Frankenstein-like sentences that are structurally perfect but contextually robotic.

## Conclusion and Next Steps
The Graph Search architecture proves that external algorithmic control can force LLMs to strictly obey formatting and lexical rules (as seen with Mistral). However, the baseline greedy search is insufficient for global constraints (Length and Tone). 

To break the 60% Strict Accuracy ceiling and optimize latency, the next iteration must implement:
1. **Length Lookahead:** A heuristic projection to penalize branches that mathematically cannot satisfy length limits given their current depth.
2. **Continuous Semantic Guidance:** Replacing the boolean end-of-generation judge with intermediate vector embeddings to guide the tree search towards the correct tone at every step.
3. **Dynamic Pruning:** Aggressively dropping bad branches early to reduce the 75-second latency cost.