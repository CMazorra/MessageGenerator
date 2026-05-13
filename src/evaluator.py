import re
import pyphen

class HaikuEvaluator:
    def __init__(self):
        # Initialize pyphen for Spanish language
        self.dic = pyphen.Pyphen(lang='es')
    
    def _count_syllables_word(self, word: str) -> int:
        """Counts the syllables of a word using pyphen."""
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        if not clean_word:
            return 0
            
        syllables = self.dic.inserted(clean_word).split('-')
        return len(syllables)

    def _count_syllables_line(self, line: str) -> int:
        """Counts total syllables in a line by summing its words."""
        words = line.split()
        total_syllables = sum(self._count_syllables_word(w) for w in words)
        return total_syllables

    def evaluate(self, generated_text: str, mandatory_word: str) -> dict:
        """
        Evaluates if the text meets the formal rules of our Haiku.
        Returns a dictionary with the evaluation details.
        """
        lines = [line.strip() for line in generated_text.strip().split('\n') if line.strip()]
        
        # 1. C1: Line structure constraint (Exactly 3 lines)
        meets_lines = len(lines) == 3
        
        # 2. C2: Metric 5-7-5 (Hard constraint)
        syllable_counts = []
        meets_metric = False
        
        if meets_lines:
            syllable_counts = [self._count_syllables_line(line) for line in lines]
            meets_metric = (syllable_counts == [5, 7, 5])
            
        # 3. C3: Content constraint (Mandatory word)
        word_regex = re.compile(rf'\b{re.escape(mandatory_word.lower())}\b')
        meets_word = bool(word_regex.search(generated_text.lower()))
        
        # Overall evaluation
        is_valid = meets_lines and meets_metric and meets_word
        
        return {
            "is_valid": is_valid,
            "meets_lines": meets_lines,
            "found_lines": len(lines),
            "meets_metric": meets_metric,
            "found_metric": syllable_counts,
            "meets_word": meets_word,
            "target_word": mandatory_word
        }

if __name__ == "__main__":
    evaluator = HaikuEvaluator()
    poem = "El código es" + "\n" + "listas en la memoria" + "\n" + "un bucle sin fin"
    result = evaluator.evaluate(poem, "listas")
    
    import json
    print(json.dumps(result, indent=2))