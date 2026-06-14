import sys
import os
import json
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from generator import DynamicGenerator
from evaluator import DynamicEvaluator

GENERATION_STRATEGY = "mcts"  # Options: 'mcts', 'search', 'direct'

def run_experiment():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'synthetic_dataset.json')
    results_path = os.path.join(base_dir, 'data', 'dynamic_experiment_results.csv')

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    generator = DynamicGenerator()
    evaluator = DynamicEvaluator()

    models_to_test = ["llama3", "mistral", "phi3"]
    # Model to act as impartial judge for the 'tone' constraint
    judge_model = "llama3" 
    
    results = []

    print(f"Starting dynamic CSP experiment with {len(dataset)} instances and {len(models_to_test)} models...")

    for model in models_to_test:
        print(f"\n--- Evaluating Generator Model: {model} ---")
        for item in tqdm(dataset, desc=f"Generating with {model}"):
            item_id = item['id']
            intent = item['intent']
            constraints = item['constraints']
            
            # Step A: Generation
            print(GENERATION_STRATEGY)
            if GENERATION_STRATEGY == 'mcts':
                generated_text, metrics = generator.generate_with_mcts(
                    intent,
                    constraints,
                    model,
                    judge_model=judge_model
                )
            elif GENERATION_STRATEGY == 'search':
                generated_text, metrics = generator.generate_with_search(
                    intent,
                    constraints,
                    model,
                    judge_model=judge_model
                )
            else:
                generated_text, metrics = generator.generate(intent, constraints, model)
            
            # Step B: Evaluation
            eval_result = evaluator.evaluate(generated_text, constraints, judge_model=judge_model)
            
            # Step C: Collect metrics
            record = {
                "instance_id": item_id,
                "generator_model": model,
                "generation_strategy": GENERATION_STRATEGY,
                "intent": intent,
                "generated_text": generated_text,
                "latency_initial_sec": metrics["initial_time"],
                "latency_retry_sec": metrics["retry_time"],
                "retries_used": metrics["retries_used"],
                "global_score": eval_result['global_score']
            }
            for k, v in metrics.items():
                if k not in {"initial_time", "retry_time", "retries_used", "final_details"}:
                    record[f"M_{k}"] = v
            # Flatten the detailed constraint evaluation into the CSV
            for k, v in eval_result['details'].items():
                record[f"C_{k}"] = v
                
            results.append(record)

    df = pd.DataFrame(results)
    df.to_csv(results_path, index=False, encoding='utf-8')
    
    print("\nExperiment finished!")
    print(f"Results saved to: {results_path}")
    
    print("\nGlobal Success Score Average (0.0 to 1.0) per model:")
    summary = df.groupby('generator_model')['global_score'].mean()
    print(summary)
    
    print("\n=== Latency and Retries per Model ===")
    latency_summary = df.groupby('generator_model')[['latency_initial_sec', 'latency_retry_sec', 'retries_used']].mean().round(3)
    print(latency_summary.to_string())

    print("\n=== Detailed Sub-Metrics per Model (%) ===")
    # 1. Strict Accuracy (Score == 1.0)
    df['strict_accuracy'] = (df['global_score'] == 1.0).astype(float)
    
    # Helper to compute row-wise means safely ignoring NaNs (blank values in CSV)
    def col_mean(cols):
        valid_cols = [c for c in cols if c in df.columns]
        return df[valid_cols].mean(axis=1) if valid_cols else float('nan')
        
    # 2. Lexical Compliance
    df['lexical_compliance'] = col_mean(['C_mandatory_words', 'C_forbidden_words'])
    # 3. Formatting Compliance
    df['formatting_compliance'] = col_mean(['C_exact_lines', 'C_format_json', 'C_required_json_keys'])
    # 4. Length Adherence
    df['length_adherence'] = col_mean(['C_min_words', 'C_max_words'])
    # 5. Semantic Success (Tone)
    df['semantic_success'] = col_mean(['C_tone'])
    
    # Group results by model, calculate the mean for each subset, round, and turn into percentage
    metrics = ['strict_accuracy', 'lexical_compliance', 'formatting_compliance', 'length_adherence', 'semantic_success']
    detailed_summary = df.groupby('generator_model')[metrics].mean().round(3) * 100
    
    print(detailed_summary.to_string())

if __name__ == "__main__":
    run_experiment()
