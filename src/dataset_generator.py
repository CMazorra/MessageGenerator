import os
import json
import re
import ollama

def generate_synthetic_dataset(num_instances: int = 20, output_path: str = "data/synthetic_dataset.json"):
    print(f"Generando {num_instances} casos de prueba sintéticos...")
    host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    client = ollama.Client(host=host)
    
    prompt_template = """
    You are an expert dataset generator for NLP constraint satisfaction problems.
    Generate exactly {count} distinct tasks. Each task must ask the user to write a short text, and must include a set of strict constraints.
    
    Vary the types of intents (e.g., emails, coding explanations, creative writing, JSON data generation).
    Vary the constraints by combining 2 to 4 of these types per task:
    - min_words (int)
    - max_words (int)
    - mandatory_words (list of strings)
    - forbidden_words (list of strings)
    - exact_lines (int)
    - format (can be "json")
    - required_json_keys (list of strings, only if format is json)
    - tone (string, e.g., "empathetic", "professional", "dramatic", "educational")

    Output ONLY a valid JSON array of objects following this exact schema:
    [
      {{
        "id": 1,
        "intent": "Write a short rejection email to a candidate.",
        "constraints": {{
          "max_words": 50,
          "forbidden_words": ["unfortunately", "sorry"],
          "tone": "formal"
        }}
      }}
    ]
    """
    
    try:
        response = client.generate(
            model="llama3",
            prompt=prompt_template.format(count=num_instances),
            options={"temperature": 0.7}
        )
        
        raw_text = response.get('response', '').strip()
        
        if '```json' in raw_text:
            match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
            if match: raw_text = match.group(1)
        elif '```' in raw_text:
            match = re.search(r'```\s*(.*?)\s*```', raw_text, re.DOTALL)
            if match: raw_text = match.group(1)
            
        dataset = json.loads(raw_text)
        
        # Guardar a disco
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        print(f"¡Éxito! Dataset de {len(dataset)} instancias guardado en {output_path}")
        
    except Exception as e:
        print(f"Error durante la generación: {e}")

if __name__ == "__main__":
    generate_synthetic_dataset(num_instances=30)