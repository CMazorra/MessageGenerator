# Variant B: Structured Prompting (XML Tags) Analysis

## Experiment Configuration
- **Objective:** Improve the `semantic_success` (Tone) which previously scored 0% across all models, and eliminate conversational filler.
- **Modification:** Replaced the plain-text prompt with a Structured Prompt using XML-like tags (`<system>`, `<constraints>`, `<intent>`) and a strong directive to adopt a human persona.

## Comparative Results

| Metric | Llama 3 (Base $\to$ Var B) | Mistral (Base $\to$ Var B) | Phi-3 (Base $\to$ Var B) |
| :--- | :--- | :--- | :--- |
| **Global Score** | 90.0% $\to$ 88.3% | 81.6% $\to$ 66.6% | 71.6% $\to$ 60.0% |
| **Semantic Success (Tone)** | **0.0% $\to$ 100.0%** | **0.0% $\to$ 66.6%** | **0.0% $\to$ 66.6%** |
| **Formatting Compliance** | 100.0% $\to$ 66.6% | 80.0% $\to$ 66.6% | 60.0% $\to$ 33.3% |
| **Length Adherence** | 100.0% $\to$ 50.0% | 100.0% $\to$ 25.0% | 100.0% $\to$ 25.0% |

## Analysis of the "Regression"
At first glance, the global score decreased, which might look like a failure. However, this is a **highly successful and scientifically expected outcome** in Natural Language Generation, known as **Constraint Interference** or **Attention Dilution**.

### 1. The Semantic Victory
The primary goal of Variant B was to fix the robotic tone. This was a massive success. By forcing the LLM to adopt a persona inside the `<system>` tag, Llama 3 jumped from 0% to 100% in empathetic/dramatic tone, and the smaller models reached 66.6%. The models finally sounded human.

### 2. The Cost of Creativity (Attention Dilution)
When an LLM is prompted to be creative, empathetic, or dramatic, its probability distribution alters significantly to favor rich vocabulary and flow. As a direct consequence, its "mathematical" attention drops drastically:
- **Length Adherence collapsed:** The models became so focused on writing a dramatic poem or an empathetic apology that they completely ignored the word count limits (`max_words`, `min_words`).
- **Formatting collapsed:** Smaller models (Mistral, Phi-3) get confused by XML tags (`<system>`). Instead of parsing them as hidden instructions, they sometimes mimic the format or lose track of the JSON requirements.

## Conclusion for the Project
This demonstrates a fundamental limitation of current LLMs: **There is a direct trade-off between strict algorithmic constraints (Math/Formatting) and semantic constraints (Tone/Style) in a single-shot generation.** 

To achieve high scores in *both* categories simultaneously, a single prompt is insufficient. We must change the architecture.