import os
import ollama

class DynamicGenerator:
    def __init__(self):
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=host)
    
    def _build_prompt(self, intent: str, constraints: dict) -> str:
        """
        Dynamically constructs the prompt based on the variable constraint set.
        """
        prompt_parts = [
            "You are an expert AI assistant that strictly follows instructions.",
            f"YOUR TASK: {intent}\n",
            "MANDATORY CONSTRAINTS YOU MUST FOLLOW PERFECTLY:"
        ]
        
        # Dynamically append constraints
        if 'min_words' in constraints:
            prompt_parts.append(f"- Your response must contain AT LEAST {constraints['min_words']} words.")
        if 'max_words' in constraints:
            prompt_parts.append(f"- Your response must contain NO MORE THAN {constraints['max_words']} words.")
        if 'exact_lines' in constraints:
            prompt_parts.append(f"- Your response must be output on EXACTLY {constraints['exact_lines']} non-empty lines.")
        if 'mandatory_words' in constraints:
            words_str = ", ".join(f"'{w}'" for w in constraints['mandatory_words'])
            prompt_parts.append(f"- You MUST include ALL of the following words verbatim: {words_str}.")
        if 'forbidden_words' in constraints:
            words_str = ", ".join(f"'{w}'" for w in constraints['forbidden_words'])
            prompt_parts.append(f"- You MUST NEVER use ANY of the following words: {words_str}.")
        if 'format' in constraints and constraints['format'] == 'json':
            prompt_parts.append("- Your response MUST be valid JSON format. Do not return Markdown, just the raw JSON object.")
        if 'required_json_keys' in constraints:
            keys_str = ", ".join(f"'{k}'" for k in constraints['required_json_keys'])
            prompt_parts.append(f"- Your JSON MUST contain exactly these top-level keys: {keys_str}.")
        if 'tone' in constraints:
            prompt_parts.append(f"- The tone of your response MUST be strictly {constraints['tone']}.")
            
        prompt_parts.append("\nReturn ONLY the generated content, without any conversational preamble or postscript.")
        
        return "\n".join(prompt_parts)
        
    def generate(self, intent: str, constraints: dict, model_name: str, temperature: float = 0.3) -> str:
        """
        Calls the LLM to generate text respecting the dynamic constraints.
        Using 0.3 temperature to balance following structural rules while allowing tone adjustments.
        """
        prompt = self._build_prompt(intent, constraints)
        
        try:
            print(f"Generating with model {model_name}...")
            response = self.client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "seed": 42
                }
            )
            raw_text = response.get('response', '').strip()
            
            # Basic cleanup for common AI chatty artifacts unless format is JSON
            if constraints.get('format') != 'json':
                lines = raw_text.split('\n')
                # If first line contains AI preamble (e.g. "Here is the poem:")
                if lines and lines[0].lower().startswith("here") and ":" in lines[0]:
                    lines = lines[1:]
                return '\n'.join(lines).strip()
            return raw_text
            
        except Exception as e:
            print(f"Error generating with {model_name}: {e}")
            return ""

if __name__ == "__main__":
    generator = DynamicGenerator()
    test_intent = "Write an apology email to a customer for a delayed package."
    test_constraints = {
        "max_words": 40,
        "mandatory_words": ["sorry", "shipping", "refund"],
        "forbidden_words": ["mistake", "fault"],
        "tone": "empathetic"
    }
    print(generator.generate(test_intent, test_constraints, "mistral"))