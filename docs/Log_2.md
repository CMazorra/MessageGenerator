# Variant C: Self-Correction (Reflection) Loop Analysis

## Experiment Configuration
- **Objective:** Mitigate the *Constraint Interference* problem observed in Variant B, where models sacrificed structural rules (length, lines, formats) to fulfill semantic tone constraints.
- **Modification:** Implemented an internal heuristic checker in Python (`_get_hard_constraint_errors`) that validates hard constraints immediately after generation. If rules are violated, an automated second prompt is sent to the LLM (with Temperature=0.1) pointing out its exact mistakes and instructing a rewrite.

## Comparative Results (Variant B $\to$ Variant C)

| Metric | Llama 3 (Var B $\to$ C) | Mistral (Var B $\to$ C) | Phi-3 (Var B $\to$ C) |
| :--- | :--- | :--- | :--- |
| **Strict Accuracy (Score=100%)** | 60.0% $\to$ **80.0%** | 40.0% $\to$ 40.0% | 20.0% $\to$ 20.0% |
| **Formatting Compliance** | 66.6% $\to$ **100.0%** | 66.6% $\to$ 66.6% | 33.3% $\to$ 33.3% |
| **Length Adherence** | 50.0% $\to$ 50.0% | 25.0% $\to$ **50.0%** | 25.0% $\to$ **75.0%** |

## Console Log Observations & Triggers
The execution logs confirm that the Reflection loop intervened actively exactly when needed:
- **Llama 3:** Triggered corrections for excessive word counts (59 words instead of 40) and line counts (4 lines instead of 3). It successfully self-corrected its formatting errors, achieving 100% Formatting Compliance and bumping its Strict Accuracy to an impressive 80%.
- **Mistral & Phi-3:** Heavily triggered the reflection loop for a wide array of rule breaks (e.g., Mistral generating 16 lines instead of 4 and hallucinating forbidden words). Interestingly, the reflection loop proved highly beneficial for Phi-3's **Length Adherence**, driving it from a dismal 25% up to 75%.

## Limitations of Single-Pass Reflection
While Llama 3 leveraged the reflection mechanism wonderfully, the smaller models faced a new bottleneck: **Catastrophic Forgetting during Editing**. When asked to "fix" a rule (like removing a forbidden word), Mistral and Phi-3 often over-edited the text, sometimes deleting mandatory words or destroying the JSON structure in the process.

This demonstrates that while *Self-Correction* is powerful, model capacity dictates its efficacy. A 3.8B parameter model (Phi-3) struggles to retain the original context while applying targeted edits, whereas an 8B model (Llama 3) handles the localized editing gracefully. 

## Final Conclusion
Variant C represents the most robust architecture developed in this project. By decoupling Semantic Generation from Algorithmic Validation, and using the latter strictly as an automated proofreader, we established a pipeline capable of generating highly constrained text with an 80% absolute perfection rate on top-tier open-weight models.