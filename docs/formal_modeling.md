# Formal Modeling: Technical Haikus

The problem consists of the automated generation of poetic messages that apply software engineering and technology concepts while respecting strict metric and structural rules.

## Hard Constraints
For a generated text to be considered valid (success), it must perfectly meet the following logical conditions $C$:

1. **Line structure ( $C_1$ )**: The text must contain exactly three (3) non-empty lines.
2. **Metric ( $C_2$ )**: 
   - The first line must have exactly 5 syllables.
   - The second line must have exactly 7 syllables.
   - The third line must have exactly 5 syllables.
3. **Content inclusion ( $C_3$ )**: The message must strictly contain a given keyword $w$ anywhere in the text.

## Soft Constraints
- **Domain and Tone**: The poem must talk about a software or technology theme specified in the input. Since these conditions are subjective, they require qualitative evaluation or the use of an LLM-as-a-judge to validate them.

## System Input
An input vector $X = [t, w]$, where:
- $t$: is the specified technical theme (e.g. "Machine Learning").
- $w$: is a mandatory word to include (e.g. "data").

## Expected Output
A text $Y$ such that eval$(Y) \to \{0,1\}$, being $1$ if it meets $C_1 \land C_2 \land C_3$, and $0$ otherwise.
