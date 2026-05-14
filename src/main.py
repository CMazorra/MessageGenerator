import sys
import os
import json
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from generator import DynamicGenerator
from evaluator import DynamicEvaluator

def run_experiment():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'dataset_constraints.json')
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
            generated_text = generator.generate(intent, constraints, model)
            
            # Step B: Evaluation
            eval_result = evaluator.evaluate(generated_text, constraints, judge_model=judge_model)
            
            # Step C: Collect metrics
            record = {
                "instance_id": item_id,
                "generator_model": model,
                "intent": intent,
                "generated_text": generated_text,
                "global_score": eval_result['global_score']
            }
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

if __name__ == "__main__":
    run_experiment()