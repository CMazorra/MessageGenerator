# Formal Modeling: Dynamic Constraint Satisfaction in NLG

The problem is conceptualized as a Natural Language Generation (NLG) task subjected to a Dynamic Constraint Satisfaction Problem (CSP). Instead of applying a fixed set of rules to all generations, the system must parse and enforce a variable subset of constraints per instance.

## Definitions

Let $I$ be the user's intent or core prompt (e.g., "Write an apology email").
Let $C = \{c_1, c_2, ..., c_k\}$ be the set of active constraints for the given intent.

The generation function is defined as:
$G(I, C) \rightarrow Y$
where $Y$ is the generated text.

## Constraint Taxonomy

Constraints are divided into two operational categories:

### 1. Hard Constraints (Algorithmic / Rule-based)
These are mathematically or logically verifiable functions $f_{hard}(Y, c_i) \rightarrow \{0, 1\}$.
Examples in our system:
- **Length Constraints**: $c_{min\_words} \le length(Y) \le c_{max\_words}$
- **Lexical Inclusion**: $\forall w \in c_{mandatory}, w \in Y$
- **Lexical Exclusion**: $\forall w \in c_{forbidden}, w \notin Y$
- **Structural Integrity**: $count\_lines(Y) == c_{exact\_lines}$
- **Format Constraints**: $is\_valid\_json(Y)$ and $contains\_keys(Y, c_{required\_keys})$

### 2. Soft Constraints (Heuristic / Semantic)
These constraints require semantic understanding to evaluate and return a continuous or discrete plausibility score $f_{soft}(Y, c_i) \rightarrow [0, 1]$.
Examples in our system:
- **Tone**: Does the text sound "empathetic", "dramatic", or "educational"?
*Note: Due to their subjective nature, Soft Constraints will be evaluated using an LLM-as-a-Judge mechanism.*

## Evaluation Metric

For a given instance with $n$ active hard constraints and $m$ active soft constraints, the **Global Success Score** ($S$) of the output block $Y$ is defined as the average satisfaction rate:

$S_{hard}(Y) = \frac{1}{n} \sum_{i=1}^{n} f_{hard}(Y, c_i)$

This allows for partial credit. A system that meets the length and inclusion constraints but fails the exclusion constraint will score $0.66$ instead of a strict boolean failure, providing better analytic granularity to compare different prompting techniques.