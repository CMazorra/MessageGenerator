import os
import ollama

class HaikuGenerator:
    def __init__(self):
        # Default to local or Docker Ollama instance
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=host)
    
    def _build_prompt(self, theme: str, mandatory_word: str) -> str:
        """Constructs a strict prompt with zero-shot configuration."""
        prompt = (
            f"You are an expert technology poet. Your task is to write a haiku in English about '{theme}'.\n\n"
            f"STRICT STRUCTURAL RULES (HARD CONSTRAINTS):\n"
            f"1. The message MUST contain EXACTLY 3 lines. No greetings, Markdown, titles, notes, or explanations.\n"
            f"2. MANDATORY syllable structure: 5 syllables in line 1, 7 syllables in line 2, and 5 syllables in line 3 (5-7-5 metric).\n"
            f"3. The word '{mandatory_word}' MUST be included verbatim anywhere in the poem.\n\n"
            f"Response (Return ONLY the poem):"
        )
        return prompt
        
    def generate_haiku(self, theme: str, mandatory_word: str, model_name: str, temperature: float = 0.1) -> str:
        """
        Calls the local LLM model to generate the haiku.
        Using low temperature for stricter compliance.
        """
        prompt = self._build_prompt(theme, mandatory_word)
        
        try:
            print(f"Generating with model {model_name}...")
            response = self.client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "seed": 42 # For deterministic results where possible
                }
            )
            # Remove whitespace and clean up
            return response.get('response', '').strip()
        except Exception as e:
            print(f"Error generating with {model_name}: {e}")
            return ""

# Basic test script limits
if __name__ == "__main__":
    generator = HaikuGenerator()
    generation = generator.generate_haiku(theme="Python", mandatory_word="lists", model_name="mistral")
    
    print("="*40)
    print("Generated Haiku:")
    print("="*40)
    print(generation)
    print("="*40)