import os
import json
import math
import random
import re
import time
import ollama

class DynamicGenerator:
    TONE_PROTOTYPES = {
        "empathetic": (
            "I am truly sorry this disrupted your day. I understand how frustrating it is to wait "
            "for something important, and I want to make this right with clear next steps."
        ),
        "educational": (
            "Think of it as a simple system with a clear purpose. First we explain the idea in plain "
            "language, then we show a small example so the concept becomes easy to remember."
        ),
        "dramatic": (
            "The room fell silent as the final clue appeared. Every choice now carried weight, and "
            "the truth seemed to wait just beyond the edge of the next breath."
        ),
        "formal": (
            "Thank you for your message. We have reviewed the matter carefully and will provide a "
            "clear, professional response with the relevant details and next steps."
        ),
        "friendly": (
            "Thanks for reaching out. I am happy to help and will keep this simple, clear, and easy "
            "to follow so we can get it sorted out together."
        )
    }

    def __init__(self, evaluator=None, semantic_scorer=None):
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        self.client = ollama.Client(host=host)
        self.evaluator = evaluator
        self.semantic_scorer = semantic_scorer
        self.embedding_model = None
    
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

    def _get_evaluator(self):
        if self.evaluator is None:
            from evaluator import DynamicEvaluator
            self.evaluator = DynamicEvaluator()
        return self.evaluator

    def _hard_constraints_only(self, constraints: dict) -> dict:
        return {key: value for key, value in constraints.items() if key != 'tone'}

    def _build_next_step_prompt(self, intent: str, constraints: dict, partial_text: str, branching_factor: int) -> str:
        output_unit = "complete valid JSON candidates" if constraints.get('format') == 'json' else "possible next sentences or lines"
        partial_block = partial_text if partial_text.strip() else "(empty)"
        return (
            "You are helping a graph-search text planner. Do not finish the explanation; only propose branches.\n\n"
            f"INTENT:\n{intent}\n\n"
            f"CONSTRAINTS:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"CURRENT PARTIAL OUTPUT:\n{partial_block}\n\n"
            f"Return exactly {branching_factor} {output_unit} that can be appended next while satisfying the constraints.\n"
            "Respond ONLY as a JSON array of strings. No markdown, no commentary."
        )

    def _build_mcts_expansion_prompt(self, intent: str, constraints: dict, partial_text: str, branching_factor: int) -> str:
        unit = "short next continuations"
        if constraints.get('format') == 'json':
            unit = "complete valid JSON candidates"
        elif 'exact_lines' in constraints:
            unit = "single next-line candidates"

        partial_block = partial_text if partial_text.strip() else "(empty)"
        tone_instruction = ""
        if 'tone' in constraints:
            tone_instruction = (
                f"\nEvery candidate must actively preserve a '{constraints['tone']}' tone. "
                "Prefer emotionally coherent wording over neutral filler."
            )
        return (
            "You are the expansion policy for Monte Carlo Tree Search over text states.\n"
            "Generate compact candidate actions only; each action must be appendable to the current partial output.\n\n"
            f"INTENT:\n{intent}\n\n"
            f"CONSTRAINTS:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"CURRENT PARTIAL OUTPUT:\n{partial_block}\n\n"
            f"{tone_instruction}\n"
            f"Return exactly {branching_factor} {unit} as a JSON array of strings. No markdown, no commentary."
        )

    def _parse_candidate_steps(self, raw_response: str, branching_factor: int) -> list[str]:
        text = raw_response.strip()
        candidates = []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                candidates = [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            candidates = []

        if not candidates:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for line in lines:
                cleaned = re.sub(r'^\s*(?:[-*]|\d+[\.)])\s*', '', line).strip()
                cleaned = cleaned.strip('"')
                if cleaned:
                    candidates.append(cleaned)

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            normalized = candidate.lower()
            if normalized not in seen:
                unique_candidates.append(candidate)
                seen.add(normalized)
            if len(unique_candidates) >= branching_factor:
                break

        return unique_candidates

    def _merge_search_step(self, partial_text: str, next_step: str, constraints: dict) -> str:
        if constraints.get('format') == 'json':
            return next_step.strip()

        partial = partial_text.strip()
        step = next_step.strip()
        if not partial:
            return step

        if 'exact_lines' in constraints:
            return f"{partial}\n{step}"
        return f"{partial} {step}"

    def _is_complete_search_state(self, text: str, constraints: dict, hard_score: float, depth: int, max_depth: int) -> bool:
        if not text.strip():
            return False
        if hard_score >= 1.0:
            return True
        if constraints.get('format') == 'json':
            return depth >= max_depth
        if 'exact_lines' in constraints:
            lines = [line for line in text.strip().split('\n') if line.strip()]
            return len(lines) >= constraints['exact_lines']
        return depth >= max_depth

    def _looks_terminal(self, text: str, constraints: dict, depth: int, max_depth: int) -> bool:
        if constraints.get('format') == 'json':
            try:
                json.loads(text.strip())
                return True
            except json.JSONDecodeError:
                return depth >= max_depth

        if 'exact_lines' in constraints:
            line_count = len([line for line in text.strip().split('\n') if line.strip()])
            return line_count >= constraints['exact_lines']

        return depth >= max_depth or text.strip().endswith(('.', '!', '?'))

    def _score_length_lookahead(self, text: str, constraints: dict, depth: int, max_depth: int) -> float:
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        remaining_depth = max(max_depth - depth, 0)
        adjustment = 0.0

        if 'max_words' in constraints:
            max_words = constraints['max_words']
            remaining_budget = max_words - word_count
            missing_mandatory = [
                word for word in constraints.get('mandatory_words', [])
                if not re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)
            ]

            if remaining_budget < 0:
                adjustment -= 0.75 + min(abs(remaining_budget) / max(max_words, 1), 1.0)
            elif remaining_budget < len(missing_mandatory):
                adjustment -= 0.50
            elif remaining_depth > 0 and not self._looks_terminal(text, constraints, depth, max_depth):
                minimum_closing_words = 3 if missing_mandatory else 2
                if remaining_budget <= minimum_closing_words:
                    adjustment -= 0.35
                elif remaining_budget <= minimum_closing_words + 2:
                    adjustment -= 0.15

        if 'min_words' in constraints:
            min_words = constraints['min_words']
            if remaining_depth == 0 and word_count < min_words:
                adjustment -= 0.40
            elif remaining_depth > 0:
                average_needed = max(min_words - word_count, 0) / remaining_depth
                if average_needed > 18:
                    adjustment -= 0.25

        return adjustment

    def _get_tone_reference(self, tone: str) -> str:
        normalized_tone = tone.strip().lower()
        return self.TONE_PROTOTYPES.get(normalized_tone, normalized_tone)

    def _lexical_tone_score(self, text: str, tone: str) -> float:
        reference_terms = set(re.findall(r'\b\w+\b', self._get_tone_reference(tone).lower()))
        text_terms = set(re.findall(r'\b\w+\b', text.lower()))
        if not reference_terms:
            return 0.0
        return len(reference_terms & text_terms) / len(reference_terms)

    def _embedding_tone_score(self, text: str, tone: str) -> float | None:
        if self.semantic_scorer is not None:
            return self.semantic_scorer.score(text, tone)

        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            return None

        # Inicialización del modelo y la caché vectorial
        if self.embedding_model is None:
            model_name = os.environ.get('SEMANTIC_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            self._tone_embeddings_cache = {} # Caché para O(1) lookups

        reference = self._get_tone_reference(tone)
        
        # Cacheamos el tensor de referencia (solo se calcula 1 vez por experimento)
        if tone not in getattr(self, '_tone_embeddings_cache', {}):
            if not hasattr(self, '_tone_embeddings_cache'):
                self._tone_embeddings_cache = {}
            self._tone_embeddings_cache[tone] = self.embedding_model.encode(reference, convert_to_tensor=True)
            
        ref_embedding = self._tone_embeddings_cache[tone]
        text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
        
        return float(util.cos_sim(text_embedding, ref_embedding).item())

    def _score_semantic_guidance(self, text: str, constraints: dict) -> float:
        tone = constraints.get('tone')
        if not tone or not text.strip():
            return 0.0

        try:
            embedding_score = self._embedding_tone_score(text, tone)
        except Exception as e:
            print(f"Semantic embedding guidance unavailable: {e}")
            embedding_score = None

        if embedding_score is None:
            return self._lexical_tone_score(text, tone)
        return max(0.0, min(embedding_score, 1.0))

    def _should_prune_state(self, text: str, constraints: dict, hard_score: float, details: dict) -> bool:
        if details.get('forbidden_words') is False:
            return True

        words = re.findall(r'\b\w+\b', text)
        if 'max_words' in constraints and len(words) > constraints['max_words']:
            return True

        if 'exact_lines' in constraints:
            line_count = len([line for line in text.strip().split('\n') if line.strip()])
            if line_count > constraints['exact_lines']:
                return True

        if hard_score < 0.25 and text.strip():
            return True

        return False

    def _is_mcts_terminal(self, text: str, constraints: dict, depth: int, max_depth: int) -> bool:
        if depth >= max_depth:
            return True

        if constraints.get('format') == 'json':
            try:
                json.loads(text.strip())
                return True
            except json.JSONDecodeError:
                return depth >= max_depth

        if 'exact_lines' in constraints:
            line_count = len([line for line in text.strip().split('\n') if line.strip()])
            return line_count >= constraints['exact_lines']

        if 'max_words' in constraints:
            word_count = len(re.findall(r'\b\w+\b', text))
            if word_count >= constraints['max_words']:
                return True

        return text.strip().endswith(('.', '!', '?')) and depth > 0

    def _build_rollout_prompt(self, text: str, intent: str, constraints: dict) -> str:
        remaining_budget = ""
        if 'max_words' in constraints:
            current_words = len(re.findall(r'\b\w+\b', text))
            remaining = max(constraints['max_words'] - current_words, 0)
            remaining_budget = f"\nYou have at most {remaining} additional words available."

        return (
            "Complete this partial message naturally for a Monte Carlo rollout.\n"
            "Keep the continuation short and fluent. Preserve the requested tone and constraints.\n"
            "Do not add explanations or markdown.\n\n"
            f"INTENT:\n{intent}\n\n"
            f"CONSTRAINTS:\n{json.dumps(constraints, ensure_ascii=False)}"
            f"{remaining_budget}\n\n"
            f"PARTIAL MESSAGE:\n{text.strip() if text.strip() else '(empty)'}\n\n"
            "Return ONLY the completed message."
        )

    def _llm_rollout_complete(
        self,
        text: str,
        intent: str,
        constraints: dict,
        model_name: str,
        rollout_temperature: float,
        seed: int
    ) -> str:
        response = self.client.generate(
            model=model_name,
            prompt=self._build_rollout_prompt(text, intent, constraints),
            options={"temperature": rollout_temperature, "seed": seed}
        )
        rollout = response.get('response', '').strip()
        return rollout or text.strip()

    def _quick_rollout_complete(self, text: str, intent: str, constraints: dict, max_depth: int) -> str:
        if constraints.get('format') == 'json':
            try:
                json.loads(text.strip())
                return text.strip()
            except json.JSONDecodeError:
                keys = constraints.get('required_json_keys', ["message"])
                mandatory_words = constraints.get('mandatory_words', [])
                description = " ".join(mandatory_words) if mandatory_words else intent
                payload = {key: description for key in keys}
                return json.dumps(payload, ensure_ascii=False)

        rollout_text = text.strip()

        if 'exact_lines' in constraints:
            target_lines = constraints['exact_lines']
            lines = [line for line in rollout_text.split('\n') if line.strip()]
            missing_words = [
                word for word in constraints.get('mandatory_words', [])
                if not re.search(rf'\b{re.escape(word)}\b', rollout_text, re.IGNORECASE)
            ]
            while len(lines) < target_lines:
                filler_words = missing_words[:2] if missing_words else ["concise", "idea"]
                lines.append(" ".join(filler_words))
                missing_words = missing_words[2:]
            return "\n".join(lines[:target_lines]).strip()

        max_words = constraints.get('max_words')
        min_words = constraints.get('min_words', 0)
        missing_words = [
            word for word in constraints.get('mandatory_words', [])
            if not re.search(rf'\b{re.escape(word)}\b', rollout_text, re.IGNORECASE)
        ]

        for missing_word in missing_words:
            words = re.findall(r'\b\w+\b', rollout_text)
            if max_words is not None and len(words) >= max_words:
                break
            if any(
                re.fullmatch(rf'{re.escape(forbidden)}', missing_word, re.IGNORECASE)
                for forbidden in constraints.get('forbidden_words', [])
            ):
                continue
            separator = "" if not rollout_text else " "
            rollout_text = f"{rollout_text}{separator}{missing_word}".strip()

        words = re.findall(r'\b\w+\b', rollout_text)
        if min_words and len(words) < min_words and 'tone' not in constraints:
            while len(words) < min_words:
                if max_words is not None and len(words) >= max_words:
                    break
                rollout_text = f"{rollout_text} message".strip()
                words = re.findall(r'\b\w+\b', rollout_text)

        if rollout_text and not rollout_text.endswith(('.', '!', '?')) and constraints.get('format') != 'json':
            rollout_text += "."

        return rollout_text

    def _mcts_reward(
        self,
        text: str,
        intent: str,
        constraints: dict,
        evaluator,
        judge_model: str,
        model_name: str,
        max_depth: int,
        semantic_guidance: bool,
        semantic_weight: float,
        rollout_policy: str,
        rollout_temperature: float,
        rollout_seed: int
    ) -> tuple[float, float, dict, str]:
        
        # --- FORZAR SIEMPRE EL ROLLOUT RÁPIDO O(1) ---
        rollout_text = self._quick_rollout_complete(text, intent, constraints, max_depth)

        hard_constraints = self._hard_constraints_only(constraints)
        if hard_constraints:
            result = evaluator.evaluate(rollout_text, hard_constraints, judge_model=judge_model)
            hard_score = result["global_score"]
            details = result["details"]
        else:
            hard_score = 1.0
            details = {}

        w_json = 0.4913
        w_length = 0.2140
        w_lexical = 0.0987
        w_semantic = 0.1961

        json_keys = [k for k in ['format_json', 'required_json_keys', 'exact_lines'] if k in details]
        score_json = sum(details[k] for k in json_keys) / len(json_keys) if json_keys else 1.0

        length_keys = [k for k in ['min_words', 'max_words'] if k in details]
        score_length_base = sum(details[k] for k in length_keys) / len(length_keys) if length_keys else 1.0
        lookahead = self._score_length_lookahead(rollout_text, constraints, max_depth, max_depth)
        score_length = max(0.0, min(1.0, score_length_base + lookahead))

        lexical_keys = [k for k in ['mandatory_words', 'forbidden_words'] if k in details]
        score_lexical = sum(details[k] for k in lexical_keys) / len(lexical_keys) if lexical_keys else 1.0

        score_semantic = 1.0
        if semantic_guidance and 'tone' in constraints:
            score_semantic = self._score_semantic_guidance(rollout_text, constraints)
            details["semantic_guidance"] = round(score_semantic, 4)

        reward = (w_json * score_json) + (w_length * score_length) + \
                 (w_lexical * score_lexical) + (w_semantic * score_semantic)
        
        reward = max(0.0, min(reward, 1.0))
        return reward, hard_score, details, rollout_text

    def _repair_tone_once(
        self,
        text: str,
        intent: str,
        constraints: dict,
        model_name: str,
        evaluator,
        judge_model: str
    ) -> tuple[str, dict, bool]:
        if 'tone' not in constraints or not text.strip():
            return text, {}, False

        original_hard = evaluator.evaluate(text, self._hard_constraints_only(constraints), judge_model=judge_model)
        original_full = evaluator.evaluate(text, constraints, judge_model=judge_model)
        if original_full["details"].get("tone") is not False:
            return text, original_full, False

        prompt = (
            "Rewrite the text to fix ONLY the tone while preserving every structural constraint.\n\n"
            f"INTENT:\n{intent}\n\n"
            f"REQUIRED TONE: {constraints['tone']}\n"
            f"CONSTRAINTS:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
            f"CURRENT TEXT:\n{text}\n\n"
            "Return ONLY the rewritten final text. Do not add explanations."
        )
        response = self.client.generate(
            model=model_name,
            prompt=prompt,
            options={"temperature": 0.2, "seed": 909}
        )
        repaired = response.get('response', '').strip()
        if not repaired:
            return text, original_full, False

        repaired_hard = evaluator.evaluate(repaired, self._hard_constraints_only(constraints), judge_model=judge_model)
        repaired_full = evaluator.evaluate(repaired, constraints, judge_model=judge_model)

        original_hard_score = original_hard["global_score"]
        repaired_hard_score = repaired_hard["global_score"]
        tone_improved = repaired_full["details"].get("tone") is True
        hard_preserved = repaired_hard_score >= original_hard_score

        if tone_improved and hard_preserved:
            return repaired, repaired_full, True
        return text, original_full, False

    def _uct_score(self, parent_visits: int, child: dict, exploration_weight: float) -> float:
        if child["visits"] == 0:
            return float('inf')
        exploitation = child["value"] / child["visits"]
        exploration = exploration_weight * math.sqrt(math.log(max(parent_visits, 1)) / child["visits"])
        return exploitation + exploration

    def _score_search_state(
        self,
        text: str,
        intent: str,
        constraints: dict,
        evaluator,
        judge_model: str,
        depth: int = 0,
        max_depth: int = 1,
        semantic_guidance: bool = True,
        semantic_weight: float = 0.15
    ) -> tuple[float, float, dict]:
        hard_constraints = self._hard_constraints_only(constraints)
        if hard_constraints:
            result = evaluator.evaluate(text, hard_constraints, judge_model=judge_model)
            hard_score = result["global_score"]
            details = result["details"]
        else:
            hard_score = 1.0
            details = {}

        w_json = 0.4913
        w_length = 0.2140
        w_lexical = 0.0987
        w_semantic = 0.1961

        json_keys = [k for k in ['format_json', 'required_json_keys', 'exact_lines'] if k in details]
        score_json = sum(details[k] for k in json_keys) / len(json_keys) if json_keys else 1.0

        length_keys = [k for k in ['min_words', 'max_words'] if k in details]
        score_length_base = sum(details[k] for k in length_keys) / len(length_keys) if length_keys else 1.0
        lookahead = self._score_length_lookahead(text, constraints, depth, max_depth)
        details["length_lookahead"] = round(lookahead, 4)
        score_length = max(0.0, min(1.0, score_length_base + lookahead))

        lexical_keys = [k for k in ['mandatory_words', 'forbidden_words'] if k in details]
        score_lexical = sum(details[k] for k in lexical_keys) / len(lexical_keys) if lexical_keys else 1.0

        score_semantic = 1.0
        if semantic_guidance and 'tone' in constraints:
            projected_text = self._quick_rollout_complete(text, intent, constraints, max_depth)
            score_semantic = self._score_semantic_guidance(projected_text, constraints)
            details["semantic_guidance"] = round(score_semantic, 4)

        search_score = (w_json * score_json) + (w_length * score_length) + \
                       (w_lexical * score_lexical) + (w_semantic * score_semantic)
        
        search_score = max(0.0, min(search_score, 1.0))
        return search_score, hard_score, details

    def generate_with_search(
        self,
        intent: str,
        constraints: dict,
        model_name: str,
        temperature: float = 0.7,
        branching_factor: int = 4,
        beam_width: int = 1,
        max_depth: int | None = None,
        judge_model: str = "llama3",
        evaluate_soft: bool = True,
        semantic_guidance: bool = True,
        semantic_weight: float = 0.15,
        pruning_threshold: float = 0.25
    ) -> tuple[str, dict]:
        """
        Generates text as a graph-search/planning problem.

        Each node is a partial output. The LLM proposes candidate next steps, and
        DynamicEvaluator scores each resulting branch on hard constraints. With
        beam_width=1 this is greedy best-first search; wider beams approximate a
        lightweight heuristic tree search.
        """
        print("Starting graph search generation...")
        evaluator = self._get_evaluator()
        if max_depth is None:
            if constraints.get('format') == 'json':
                max_depth = 1
            elif 'exact_lines' in constraints:
                max_depth = constraints['exact_lines']
            else:
                max_depth = 4

        start_t = time.time()
        metrics = {
            "initial_time": 0.0,
            "retry_time": 0.0,
            "retries_used": 0,
            "algorithm": "heuristic_graph_search",
            "branching_factor": branching_factor,
            "beam_width": beam_width,
            "search_depth": 0,
            "nodes_expanded": 0,
            "candidates_evaluated": 0,
            "candidates_pruned": 0,
            "semantic_guidance": semantic_guidance,
            "semantic_weight": semantic_weight,
            "pruning_threshold": pruning_threshold,
            "best_hard_score": 0.0
        }

        frontier = [{"text": "", "score": 0.0, "hard_score": 0.0, "details": {}}]
        best_state = frontier[0]

        try:
            for depth in range(1, max_depth + 1):
                expanded_states = []

                for state in frontier:
                    prompt = self._build_next_step_prompt(intent, constraints, state["text"], branching_factor)
                    response = self.client.generate(
                        model=model_name,
                        prompt=prompt,
                        options={"temperature": temperature, "seed": 42 + depth}
                    )
                    metrics["nodes_expanded"] += 1

                    candidates = self._parse_candidate_steps(response.get('response', ''), branching_factor)
                    for candidate in candidates:
                        candidate_text = self._merge_search_step(state["text"], candidate, constraints)
                        score, hard_score, details = self._score_search_state(
                            candidate_text,
                            intent,
                            constraints,
                            evaluator,
                            judge_model,
                            depth=depth,
                            max_depth=max_depth,
                            semantic_guidance=semantic_guidance,
                            semantic_weight=semantic_weight
                        )
                        metrics["candidates_evaluated"] += 1
                        if self._should_prune_state(candidate_text, constraints, hard_score, details) or score < pruning_threshold:
                            metrics["candidates_pruned"] += 1
                            continue

                        expanded_states.append({
                            "text": candidate_text,
                            "score": score,
                            "hard_score": hard_score,
                            "details": details
                        })

                if not expanded_states:
                    break

                expanded_states.sort(key=lambda item: (item["score"], item["hard_score"]), reverse=True)
                frontier = expanded_states[:max(1, beam_width)]
                best_state = frontier[0]
                metrics["search_depth"] = depth
                metrics["best_hard_score"] = best_state["hard_score"]

                if self._is_complete_search_state(best_state["text"], constraints, best_state["hard_score"], depth, max_depth):
                    break

            final_text = best_state["text"].strip()
            if evaluate_soft and 'tone' in constraints and final_text:
                final_result = evaluator.evaluate(final_text, constraints, judge_model=judge_model)
                metrics["final_global_score"] = final_result["global_score"]
                metrics["final_details"] = final_result["details"]

            metrics["initial_time"] = time.time() - start_t
            return final_text, metrics

        except Exception as e:
            print(f"Error generating with graph search using {model_name}: {e}")
            metrics["initial_time"] = time.time() - start_t
            return best_state["text"].strip(), metrics

    def generate_with_mcts(
        self,
        intent: str,
        constraints: dict,
        model_name: str,
        temperature: float = 0.8,
        iterations: int = 24,
        branching_factor: int = 2,
        max_depth: int | None = None,
        exploration_weight: float = 1.414,
        judge_model: str = "llama3",
        evaluate_soft: bool = True,
        semantic_guidance: bool = True,
        semantic_weight: float = 0.70,
        tone_repair: bool = True,
        rollout_policy: str = "llm",
        rollout_temperature: float = 0.9
    ) -> tuple[str, dict]:
        """
        Generates text with Monte Carlo Tree Search.

        The LLM is the expansion policy. A cheap rollout policy completes partial
        states, DynamicEvaluator scores simulated terminal messages, and rewards
        are backpropagated through the tree with UCB1/UCT selection.
        """
        print("Starting MCTS generation...")
        evaluator = self._get_evaluator()
        if max_depth is None:
            if constraints.get('format') == 'json':
                max_depth = 1
            elif 'exact_lines' in constraints:
                max_depth = constraints['exact_lines']
            else:
                max_depth = 4

        start_t = time.time()
        metrics = {
            "initial_time": 0.0,
            "retry_time": 0.0,
            "retries_used": 0,
            "algorithm": "mcts",
            "mcts_iterations": iterations,
            "branching_factor": branching_factor,
            "max_depth": max_depth,
            "exploration_weight": exploration_weight,
            "nodes_expanded": 0,
            "rollouts": 0,
            "best_reward": 0.0,
            "best_hard_score": 0.0,
            "semantic_guidance": semantic_guidance,
            "semantic_weight": semantic_weight,
            "tone_repair": tone_repair,
            "tone_repair_used": 0,
            "rollout_policy": rollout_policy,
            "rollout_temperature": rollout_temperature
        }

        def new_node(text: str, parent=None, depth: int = 0) -> dict:
            return {
                "text": text.strip(),
                "parent": parent,
                "children": [],
                "untried_actions": [],
                "expanded": False,
                "visits": 0,
                "value": 0.0,
                "depth": depth,
                "best_rollout": text.strip(),
                "best_reward": 0.0,
                "hard_score": 0.0,
                "details": {}
            }

        root = new_node("")
        best_node = root

        try:
            for iteration in range(iterations):
                node = root
                path = [node]

                while (
                    node["expanded"]
                    and not node["untried_actions"]
                    and node["children"]
                    and not self._is_mcts_terminal(node["text"], constraints, node["depth"], max_depth)
                ):
                    node = max(
                        node["children"],
                        key=lambda child: self._uct_score(node["visits"], child, exploration_weight)
                    )
                    path.append(node)

                if not self._is_mcts_terminal(node["text"], constraints, node["depth"], max_depth):
                    if not node["expanded"]:
                        prompt = self._build_mcts_expansion_prompt(
                            intent,
                            constraints,
                            node["text"],
                            branching_factor
                        )
                        response = self.client.generate(
                            model=model_name,
                            prompt=prompt,
                            options={"temperature": temperature, "seed": 1000 + iteration}
                        )
                        actions = self._parse_candidate_steps(response.get('response', ''), branching_factor)
                        node["untried_actions"] = actions
                        node["expanded"] = True
                        metrics["nodes_expanded"] += 1

                    if node["untried_actions"]:
                        action = node["untried_actions"].pop(0)
                        child_text = self._merge_search_step(node["text"], action, constraints)
                        child = new_node(child_text, parent=node, depth=node["depth"] + 1)
                        node["children"].append(child)
                        node = child
                        path.append(node)

                reward, hard_score, details, rollout_text = self._mcts_reward(
                    node["text"],
                    intent,
                    constraints,
                    evaluator,
                    judge_model,
                    model_name,
                    max_depth,
                    semantic_guidance,
                    semantic_weight,
                    rollout_policy,
                    rollout_temperature,
                    2000 + iteration
                )
                metrics["rollouts"] += 1

                for visited in path:
                    visited["visits"] += 1
                    visited["value"] += reward
                    if reward > visited["best_reward"]:
                        visited["best_reward"] = reward
                        visited["best_rollout"] = rollout_text
                        visited["hard_score"] = hard_score
                        visited["details"] = details

                if reward > best_node["best_reward"]:
                    best_node = node
                    metrics["best_reward"] = reward
                    metrics["best_hard_score"] = hard_score

            if root["children"]:
                best_child = max(
                    root["children"],
                    key=lambda child: (
                        child["best_reward"],
                        child["value"] / child["visits"] if child["visits"] else 0.0,
                        child["visits"]
                    )
                )
                final_text = best_child["best_rollout"].strip()
                metrics["best_reward"] = best_child["best_reward"]
                metrics["best_hard_score"] = best_child["hard_score"]
                metrics["root_children"] = len(root["children"])
            else:
                final_text = best_node["best_rollout"].strip()
                metrics["root_children"] = 0

            if evaluate_soft and 'tone' in constraints and final_text:
                if tone_repair:
                    final_text, final_result, repair_used = self._repair_tone_once(
                        final_text,
                        intent,
                        constraints,
                        model_name,
                        evaluator,
                        judge_model
                    )
                    metrics["tone_repair_used"] = int(repair_used)
                else:
                    final_result = evaluator.evaluate(final_text, constraints, judge_model=judge_model)
                metrics["final_global_score"] = final_result["global_score"]
                metrics["final_details"] = final_result["details"]

            metrics["initial_time"] = time.time() - start_t
            return final_text, metrics

        except Exception as e:
            print(f"Error generating with MCTS using {model_name}: {e}")
            metrics["initial_time"] = time.time() - start_t
            return best_node["best_rollout"].strip(), metrics

    def generate(self, intent: str, constraints: dict, model_name: str, temperature: float = 0.3, max_retries: int = 1) -> tuple[str, dict]:
        """
        Calls the LLM with Self-Correction (Reflection loop) if hard constraints fail.
        Returns (final_text, latency_metrics).
        """
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
