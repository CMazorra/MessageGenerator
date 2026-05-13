# Baseline Results Analysis (Zero-Shot)

## Experiment Configuration
- **Dataset:** 5 instances (Themes: Python, Machine Learning, Databases, Recursion, Version Control).
- **Models tested:** Llama 3 (8B), Mistral (7B), Phi-3 (3.8B).
- **Prompting Technique:** Zero-Shot (Direct instruction without examples).
- **Temperature:** 0.1 (Low temperature to encourage deterministic and rule-following behavior).

## Global Results
The initial experiment using a strict Zero-Shot approach yielded a **0% Global Success Rate (Constraint Satisfaction Rate)** across all three models. None of the models were able to successfully generate a text that satisfied $C_1 \land C_2 \land C_3$ simultaneously.

## Detailed Failure Analysis

Based on the `experiment_results.csv` data, failures can be categorized into three main issues:

### 1. The Token vs. Syllable Problem ($C_2$ Failure)
The most common point of failure was the $C_2$ constraint (5-7-5 metric). 
LLMs process text through sub-word *tokens*, not phonetic sounds or characters. For an LLM to count syllables, it must rely on its pre-trained statistical correlation rather than mathematical phonetic analysis. As a result, when asked for 5-7-5, the models frequently output erratic patterns (e.g., 4-8-6, 5-9-5).

### 2. The "Chatty Assistant" Syndrome ($C_1$ Failure)
Despite the hard constraint explicitly stating "Return ONLY the poem" and "No greetings", small instruction-tuned models (especially Mistral and Phi-3) struggle to suppress their conversational alignment.
Many outputs included preamble or post-generation remarks, such as:
- *"Here is your haiku:"*
- *(The actual 3-line poem)*
- *"Hope you find this helpful!"*

This immediately breaks constraint $C_1$, which dictates the string must contain exactly 3 lines (`len(lines) == 3`).

### 3. Keyword Omission ($C_3$ Failure)
In fewer cases, while attempting to force the rhyme or the structural metric, the LLM hallucinates or drops the mandatory word $w$. By focusing heavily on the syllable counting mechanism, the attention mechanism overlooks the simple inclusion of the requested term.

## Next Steps and Proposed Variations

The proposed improvement involves modifying the Generation Algorithm:
1. **Prompt Engineering Upgrade:** Transition from *Zero-Shot* to **Few-Shot Prompting**. By providing the LLM with a verified example of an accepted Haiku, we can implicitly teach it the exact spatial logic of 3 lines without conversational filler.
2. **Chain-of-Thought (Optional):** Asking the model to "count the syllables out loud" before printing the final poem, forcing its attention mechanism to process the metric before outputting the final string.
3. **Regex Cleanup (Post-processing):** Implement basic text sanitation in `generator.py` to strip out common conversational filler (e.g., removing lines that end in colons `:` or contain the word "Here is").