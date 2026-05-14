import os
import ollama

class DynamicGenerator:
    def __init__(self):
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=host)
    
    def _build_prompt(self, intent: str, constraints: dict) -> str:
        """
        Variant B: Structured Prompting with XML-like tags and Enhanced Persona.
        Reduces AI "chattiness" and enforces stricter attention to soft constraints.
        """
        prompt_parts = [
            "<system>",
            "You are an advanced text generation engine. Your primary function is to satisfy structural and semantic constraints perfectly. You have no personality of your own; you must completely adopt the requested persona and tone.",
            "NEVER include conversational filler (e.g., 'Here is your text', 'Sure!'). NEVER break the requested format.",
            "</system>\n",
            "<constraints>"
        ]
        
        # Dynamically append constraints
        if 'min_words' in constraints:
            prompt_parts.append(f"- LENGTH: At least {constraints['min_words']} words.")
        if 'max_words' in constraints:
            prompt_parts.append(f"- LENGTH: Maximum {constraints['max_words']} words.")
        if 'exact_lines' in constraints:
            prompt_parts.append(f"- STRUCTURE: Exactly {constraints['exact_lines']} non-empty lines.")
        if 'mandatory_words' in constraints:
            words_str = ", ".join(f"'{w}'" for w in constraints['mandatory_words'])
            prompt_parts.append(f"- LEXICAL INCLUSION: MUST contain ALL these words exactly: {words_str}.")
        if 'forbidden_words' in constraints:
            words_str = ", ".join(f"'{w}'" for w in constraints['forbidden_words'])
            prompt_parts.append(f"- LEXICAL EXCLUSION: MUST NEVER contain ANY of these words: {words_str}.")
        if 'format' in constraints and constraints['format'] == 'json':
            prompt_parts.append("- FORMAT: Valid JSON only. NO Markdown wrappers. NO backticks.")
        if 'required_json_keys' in constraints:
            keys_str = ", ".join(f"'{k}'" for k in constraints['required_json_keys'])
            prompt_parts.append(f"- JSON SCHEMA: Must include exactly these keys: {keys_str}.")
        if 'tone' in constraints:
            prompt_parts.append(f"- SEMANTICS/TONE: You must sound strictly '{constraints['tone']}'. Do not sound robotic, formal, or like an AI unless requested. Be deeply human if the tone requires it (e.g., empathetic).")
            
        prompt_parts.append("</constraints>\n")
        
        prompt_parts.append("<intent>")
        prompt_parts.append(intent)
        prompt_parts.append("</intent>\n")
        
        # Strict stopping cue
        prompt_parts.append("FINAL OUTPUT (Just the requested text):")
        
        return "\n".join(prompt_parts)
        
    def _get_hard_constraint_errors(self, text: str, constraints: dict) -> list:
        """Internal lightweight checker to find hard constraint violations for reflection."""
        import re, json
        errors = []
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        if 'min_words' in constraints and word_count < constraints['min_words']:
            errors.append(f"Text has {word_count} words but minimum is {constraints['min_words']}.")
        if 'max_words' in constraints and word_count > constraints['max_words']:
            errors.append(f"Text has {word_count} words but maximum is {constraints['max_words']}.")
        if 'exact_lines' in constraints:
            lines = [line for line in text.split('\n') if line.strip()]
            if len(lines) != constraints['exact_lines']:
                errors.append(f"Text has {len(lines)} non-empty lines but must have exactly {constraints['exact_lines']}.")
                
        if 'mandatory_words' in constraints:
            missing = [w for w in constraints['mandatory_words'] if not re.search(rf'\b{re.escape(w)}\b', text, re.IGNORECASE)]
            if missing: errors.append(f"Missing mandatory words: {', '.join(missing)}.")
            
        if 'forbidden_words' in constraints:
            found = [w for w in constraints['forbidden_words'] if re.search(rf'\b{re.escape(w)}\b', text, re.IGNORECASE)]
            if found: errors.append(f"Used forbidden words: {', '.join(found)}.")
            
        if 'format' in constraints and constraints['format'] == 'json':
            clean_text = text.strip()
            if '```json' in clean_text:
                match = re.search(r'```json\s*(.*?)\s*```', clean_text, re.DOTALL)
                if match: clean_text = match.group(1)
            elif '```' in clean_text:
                match = re.search(r'```\s*(.*?)\s*```', clean_text, re.DOTALL)
                if match: clean_text = match.group(1)
                
            try:
                parsed = json.loads(clean_text)
                if 'required_json_keys' in constraints:
                    missing_keys = [k for k in constraints['required_json_keys'] if k not in parsed]
                    if missing_keys: errors.append(f"JSON is missing required keys: {', '.join(missing_keys)}.")
            except json.JSONDecodeError:
                errors.append("Output is not validly parseable JSON.")
                
        return errors

    def generate(self, intent: str, constraints: dict, model_name: str, temperature: float = 0.3, max_retries: int = 1) -> tuple[str, dict]:
        """
        Calls the LLM with Self-Correction (Reflection loop) if hard constraints fail.
        Returns (final_text, latency_metrics).
        """
        import time
        prompt = self._build_prompt(intent, constraints)
        
        metrics = {"initial_time": 0.0, "retry_time": 0.0, "retries_used": 0}
        
        try:
            print(f"Generating draft with model {model_name}...")
            start_t = time.time()
            response = self.client.generate(model=model_name, prompt=prompt, options={"temperature": temperature, "seed": 42})
            metrics["initial_time"] = time.time() - start_t
            draft_text = response.get('response', '').strip()
            
            # Autocorrection Loop
            retry_start_t = time.time()
            for attempt in range(max_retries):
                errors = self._get_hard_constraint_errors(draft_text, constraints)
                if not errors:
                    break # Perfect generation, break the loop
                
                metrics["retries_used"] += 1
                print(f"[{model_name}] Self-Correction triggered. Errors found: {errors}")
                reflection_prompt = (
                    f"You generated the following text based on my instructions:\n\"\"\"{draft_text}\"\"\"\n\n"
                    f"However, your text FAILED the following structural constraints:\n" +
                    "\n".join([f"- {err}" for err in errors]) +
                    f"\n\nPlease REWRITE the text to fix these errors while keeping the requested tone and subject intact.\n"
                    f"Return ONLY the fixed content."
                )
                
                # Ask the model to correct its own mistake
                correction_response = self.client.generate(model=model_name, prompt=reflection_prompt, options={"temperature": 0.1}) # Lower temp for fixing
                draft_text = correction_response.get('response', '').strip()

            metrics["retry_time"] = time.time() - retry_start_t
            
            # Final cleanup
            if constraints.get('format') != 'json':
                lines = draft_text.split('\n')
                if lines and lines[0].lower().startswith("here") and ":" in lines[0]:
                    lines = lines[1:]
                return '\n'.join(lines).strip(), metrics
            return draft_text, metrics
            
        except Exception as e:
            print(f"Error generating with {model_name}: {e}")
            return "", metrics

if __name__ == "__main__":
    generator = DynamicGenerator()
    test_intent = "Write an apology email to a customer for a delayed package."
    test_constraints = {
        "max_words": 40,
        "mandatory_words": ["sorry", "shipping", "refund"],
        "forbidden_words": ["mistake", "fault"],
        "tone": "empathetic"
    }
    text, metrics = generator.generate(test_intent, test_constraints, "mistral")
    print("\n[RESULT]")
    print(text)
    print("\n[METRICS]", metrics)
