import re
import json
import os
import ollama

class DynamicEvaluator:
    def __init__(self):
        # We initialize the client to allow LLM-as-a-judge capabilities for Soft Constraints
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=host)

    def _get_word_count(self, text: str) -> int:
        """Helper to count actual words, ignoring punctuation."""
        return len(re.findall(r'\b\w+\b', text))

    def _evaluate_tone_with_llm(self, text: str, expected_tone: str, judge_model: str) -> bool:
        """
        Soft Constraint: Uses an LLM to evaluate if the text meets the expected tone.
        """
        prompt = (
            f"You are an expert impartial judge evaluating the tone of a text.\n\n"
            f"TEXT TO EVALUATE:\n\"\"\"{text}\"\"\"\n\n"
            f"QUESTION: Does the text successfully convey a strictly '{expected_tone}' tone?\n"
            f"Respond ONLY with YES or NO."
        )
        try:
            # We use temperature 0.0 for maximum consistency in the judge
            response = self.client.generate(model=judge_model, prompt=prompt, options={"temperature": 0.0})
            answer = response.get('response', '').strip().upper()
            print(f"LLM Judge Response for tone evaluation: '{answer}'")
            # Simple check: Does the LLM judge's text start with YES or contain YES prominently?
            if answer.startswith('YES') or 'YES,' in answer or answer == 'YES.':
                return True
            return False
        except Exception as e:
            print(f"Error in LLM Judge [{judge_model}]: {e}")
            return False

    def evaluate(self, generated_text: str, constraints: dict, judge_model: str = "llama3") -> dict:
        """
        Dynamically applies all constraints found in the dictionary to the given text.
        Returns a detailed map of passed constraints and a Global Success Score between 0.0 and 1.0.
        """
        results = {}
        passed_count = 0
        total_count = len(constraints)
        
        if total_count == 0:
            return {"global_score": 1.0, "details": {}}

        # --- HARD CONSTRAINTS ---

        if 'min_words' in constraints:
            passed = self._get_word_count(generated_text) >= constraints['min_words']
            results['min_words'] = passed
            passed_count += int(passed)

        if 'max_words' in constraints:
            passed = self._get_word_count(generated_text) <= constraints['max_words']
            results['max_words'] = passed
            passed_count += int(passed)

        if 'exact_lines' in constraints:
            lines = [line for line in generated_text.strip().split('\n') if line.strip()]
            passed = len(lines) == constraints['exact_lines']
            results['exact_lines'] = passed
            passed_count += int(passed)

        if 'mandatory_words' in constraints:
            req_words = constraints['mandatory_words']
            passed = all(re.search(rf'\b{re.escape(w)}\b', generated_text, re.IGNORECASE) for w in req_words)
            results['mandatory_words'] = passed
            passed_count += int(passed)

        if 'forbidden_words' in constraints:
            forb_words = constraints['forbidden_words']
            passed = not any(re.search(rf'\b{re.escape(w)}\b', generated_text, re.IGNORECASE) for w in forb_words)
            results['forbidden_words'] = passed
            passed_count += int(passed)

        # JSON Formatting evaluation
        parsed_json = None
        if 'format' in constraints and constraints['format'] == 'json':
            clean_text = generated_text.strip()
            # Clean possible markdown blocks typically produced by LLMs
            if '```json' in clean_text:
                match = re.search(r'```json\s*(.*?)\s*```', clean_text, re.DOTALL)
                if match: clean_text = match.group(1)
            elif '```' in clean_text:
                match = re.search(r'```\s*(.*?)\s*```', clean_text, re.DOTALL)
                if match: clean_text = match.group(1)

            try:
                parsed_json = json.loads(clean_text)
                passed = True
            except json.JSONDecodeError:
                passed = False
                
            results['format_json'] = passed
            passed_count += int(passed)

        if 'required_json_keys' in constraints:
            if parsed_json and isinstance(parsed_json, dict):
                passed = all(k in parsed_json for k in constraints['required_json_keys'])
            else:
                passed = False
            results['required_json_keys'] = passed
            passed_count += int(passed)

        # --- SOFT CONSTRAINTS ---
        
        if 'tone' in constraints:
            passed = self._evaluate_tone_with_llm(generated_text, constraints['tone'], judge_model)
            results['tone'] = passed
            passed_count += int(passed)

        # Calculate partial success representation
        score = passed_count / total_count

        return {
            "global_score": score,
            "details": results
        }

if __name__ == "__main__":
    # Quick Test
    evaluator = DynamicEvaluator()
    sample_text = "I am very sorry for the shipping delay. We will process your refund."
    sample_constraints = {
        "max_words": 40,
        "mandatory_words": ["sorry", "shipping", "refund"],
        "forbidden_words": ["mistake", "fault"],
        "tone": "empathetic"
    }
    # This might take a second as it spins up a local model for the soft constraint
    print("Testing Evaluator...")
    res = evaluator.evaluate(sample_text, sample_constraints, judge_model="llama3")
    print(json.dumps(res, indent=2))