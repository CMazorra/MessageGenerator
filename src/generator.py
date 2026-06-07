import os
import json
import re
import time
import ollama

class DynamicGenerator:
    TONE_PROTOTYPES = {
        "empathetic": "warm compassionate understanding sorry reassuring human supportive caring",
        "educational": "clear simple explanatory patient helpful instructive approachable",
        "dramatic": "intense vivid emotional suspenseful poetic urgent expressive",
        "formal": "professional respectful concise structured polished courteous",
        "friendly": "warm casual positive approachable cheerful conversational"
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

        if self.embedding_model is None:
            model_name = os.environ.get('SEMANTIC_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)

        reference = self._get_tone_reference(tone)
        embeddings = self.embedding_model.encode([text, reference], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]).item())

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

    def _score_search_state(
        self,
        text: str,
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

        words = re.findall(r'\b\w+\b', text)
        search_score = hard_score

        if 'mandatory_words' in constraints and constraints['mandatory_words']:
            found = sum(
                1 for word in constraints['mandatory_words']
                if re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)
            )
            search_score += 0.10 * (found / len(constraints['mandatory_words']))

        if 'min_words' in constraints and constraints['min_words'] > 0:
            search_score += 0.05 * min(len(words) / constraints['min_words'], 1.0)

        if 'exact_lines' in constraints and constraints['exact_lines'] > 0:
            line_count = len([line for line in text.strip().split('\n') if line.strip()])
            search_score += 0.05 * min(line_count / constraints['exact_lines'], 1.0)
            if line_count > constraints['exact_lines']:
                search_score -= 0.25

        if 'max_words' in constraints and len(words) > constraints['max_words']:
            search_score -= 0.25

        if 'forbidden_words' in constraints:
            has_forbidden = any(
                re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)
                for word in constraints['forbidden_words']
            )
            if has_forbidden:
                search_score -= 0.50

        lookahead_adjustment = self._score_length_lookahead(text, constraints, depth, max_depth)
        search_score += lookahead_adjustment
        details["length_lookahead"] = round(lookahead_adjustment, 4)

        if semantic_guidance and 'tone' in constraints:
            semantic_score = self._score_semantic_guidance(text, constraints)
            search_score += semantic_weight * semantic_score
            details["semantic_guidance"] = round(semantic_score, 4)

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
