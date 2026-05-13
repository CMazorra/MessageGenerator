import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# Ensure Python can find local modules
sys.path.append(os.path.dirname(__file__))

from generator import HaikuGenerator
from evaluator import HaikuEvaluator

def run_experiment():
    # Setup relative paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'dataset_haikus.json')
    results_path = os.path.join(base_dir, 'data', 'experiment_results.csv')

    # Load Dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    generator = HaikuGenerator()
    evaluator = HaikuEvaluator()
    
    # Local models downloaded via Docker
    models_to_test = ["llama3", "mistral", "phi3"]
    results = []

    print(f"Starting experiment with {len(dataset)} instances and {len(models_to_test)} models...")

    for model in models_to_test:
        print(f"\n--- Evaluating model: {model} ---")
        # Using tqdm for a console progress bar
        for item in tqdm(dataset, desc=f"Generating with {model}"):
            item_id = item['id']
            theme = item['theme']
            word = item['mandatory_word']
            
            # Step A: Generation
            generated_text = generator.generate_haiku(theme, word, model)
            
            # Step B: Evaluation
            eval_result = evaluator.evaluate(generated_text, word)
            
            # Step C: Combined result collection
            record = {
                "instance_id": item_id,
                "theme": theme,
                "mandatory_word": word,
                "model": model,
                "generated_text": generated_text,
                "is_valid_global": eval_result['is_valid'],
                "meets_lines": eval_result['meets_lines'],
                "found_lines": eval_result['found_lines'],
                "meets_metric": eval_result['meets_metric'],
                "found_metric": str(eval_result['found_metric']), # Saved as string for CSV
                "meets_word": eval_result['meets_word']
            }
            results.append(record)

    # Export results with Pandas for easy analysis
    df = pd.DataFrame(results)
    df.to_csv(results_path, index=False, encoding='utf-8')
    
    print("\nExperiment finished!")
    print(f"Detailed results have been saved to: {results_path}")
    
    # Calculate and print a quick summary
    print("\nSuccess Rate (%) per model (Constraint Satisfaction Rate):")
    summary = df.groupby('model')['is_valid_global'].mean() * 100
    print(summary)

if __name__ == "__main__":
    run_experiment()