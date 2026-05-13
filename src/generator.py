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
            f"Eres un experto poeta de tecnología. Tu tarea es escribir un haiku en español sobre '{theme}'.\n\n"
            f"REGLAS ESTRUCTURALES ESTRICTAS OBLIGATORIAS:\n"
            f"1. El mensaje debe contener EXACTAMENTE 3 líneas. Nada de saludos, Markdown, títulos, notas ni explicaciones.\n"
            f"2. Estructura silábica OBLIGATORIA: 5 sílabas en la línea 1, 7 sílabas en la línea 2, y 5 sílabas en la línea 3 (métrica 5-7-5).\n"
            f"3. La palabra '{mandatory_word}' DEBE estar incluida textualmente en cualquier lugar del poema.\n\n"
            f"Respuesta (Devuelve ÚNICAMENTE el poema):"
        )
        return prompt
        
    def generate_haiku(self, theme: str, mandatory_word: str, model_name: str, temperature: float = 0.1) -> str:
        """
        Calls the local LLM model to generate the haiku.
        Using low temperature for stricter compliance.
        """
        prompt = self._build_prompt(theme, mandatory_word)
        
        try:
            print(f"Generando con modelo {model_name}...")
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
            print(f"Error generando con {model_name}: {e}")
            return ""

# Basic test script limits
if __name__ == "__main__":
    generator = HaikuGenerator()
    generation = generator.generate_haiku(theme="Python", mandatory_word="listas", model_name="mistral")
    
    print("="*40)
    print("Haiko Generado:")
    print("="*40)
    print(generation)
    print("="*40)