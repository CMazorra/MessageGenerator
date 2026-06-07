# Dynamic Constraint Satisfaction: Initial Results

## Experiment Configuration
- **Dataset:** 5 dynamic instances representing varied real-world intents (Apology email, Simplification for children, JSON Product Description, Dramatic Poem, Tech Startup Ideas).
- **Generator Models:** Llama 3 (8B), Mistral (7B), Phi-3 (3.8B).
- **Judge Model (Soft Constraints):** Llama 3 (Temperature: 0.0) acting as an impartial evaluator for "Tone".
- **Constraints Applied dynamically:** Length bounds (`min_words`, `max_words`), Lexical (`mandatory_words`, `forbidden_words`), Formatting (`format: json`, `required_json_keys`), Structural (`exact_lines`), and Semantic (`tone`).

## Global Success Scores
The metric calculated is the average Constraint Satisfaction Rate (from $0.0$ to $1.0$) across all instances.

| Generator Model | Average Global Score |
| :--- | :--- |
| **Llama 3 (8B)** | **0.900** (90%) |
| **Mistral (7B)** | **0.816** (81.6%) |
| **Phi-3 (3.8B)** | **0.716** (71.6%) |

## Detailed Analysis and Observations

Based on the `dynamic_experiment_results.csv` data and the execution logs, we can draw the following conclusions regarding the NLG capabilities under dynamic restrictions:

### 1. Model Size vs. Constraint Adherence
The hierarchy mathematically validates that larger parameter models are inherently better at attending to multiple simultaneous rules. Llama 3 achieved near-perfect formatting, only missing points on the subjective tone evaluation. Phi-3, being the smallest, struggled with multi-variable instructions.

### 2. The Vulnerability of JSON and Formatting (Phi-3)
Instance 3 required a strict JSON format. While Llama 3 and Mistral returned clean dictionaries, **Phi-3 enclosed the output in Markdown blocks** (````json ... ````). Fortunately, our `evaluator.py` included a resilient regex to extract the JSON, but it highlights how smaller models are heavily biased towards conversational Markdown, struggling to output raw programmatic text natively.

### 3. "Chatty Assistant" Relapse
Instance 1 and 5 showed that Phi-3 still appends conversational text. For example, it generated:
> *"Here'supperly more challenging instruction with at least 5 additional constraints:"*
This uncontrollable hallucinated postscript automatically breaks constraints like `exact_lines`, dropping its score significantly.

### 4. LLM-as-a-Judge: The Empathy Deficit
The most interesting data point comes from the Soft Constraint evaluation. For Instance 1 ("Write an apology email... tone: empathetic"), **the Llama-3 Judge evaluated all three models as 'NO' (Non-empathetic)**. 
Despite the generators including the mandatory words ("sorry", "refund", "shipping") and respecting the word limits, the resulting texts felt overly corporate, robotic, and transactional (e.g., *"Dear Customer... We understand this inconvenience"*). The impartial judge correctly penalized this, proving that satisfying Hard Constraints (lexical inclusion) does not automatically guarantee semantic success.

## Next Steps
To push the models closer to a $1.0$ (100%) success rate, the following solutions can be explored:
1. **System Prompts vs User Prompts:** Shift the constraints exclusively to a `System` role message so the model doesn't treat them as a continuous chat conversation.
2. **Output Enforcement:** Implement strict stopping criteria (stop words) or formal output schemas (like OpenAI's structured outputs / Grammar-constrained decoding) to prevent Phi-3's chatty regressions.
3. **Tone Injection:** Prompt the models to use empathetic psychological frameworks rather than just requesting an "empathetic tone".