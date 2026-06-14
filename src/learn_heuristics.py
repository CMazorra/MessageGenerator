import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def optimize_heuristic_weights(csv_path=None):
    print("--- Optimizador de Heurísticas (Machine Learning) ---")
    
    data = {
        'score_json': [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0],
        'score_length': [1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        'score_lexical': [1.0, 1.0, 0.8, 1.0, 0.5, 1.0, 0.9, 0.8, 0.2, 1.0],
        'score_semantic': [0.9, 0.8, 0.7, 0.9, 0.4, 0.9, 0.6, 0.8, 0.1, 0.9],
        'success': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1] 
    }
    df = pd.DataFrame(data)
    
    X = df[['score_json', 'score_length', 'score_lexical', 'score_semantic']]
    y = df['success']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    raw_weights = model.coef_[0]
    positive_weights = np.maximum(0, raw_weights) 
    
    if np.sum(positive_weights) == 0:
        normalized_weights = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        normalized_weights = positive_weights / np.sum(positive_weights)
    
    print("\n✅ Entrenamiento completado (con escalado de características).")
    print("\n--- Nuevos Pesos Heurísticos Optimizados ---")
    print(f"w_json (Formato):    {normalized_weights[0]:.4f}")
    print(f"w_length (Longitud): {normalized_weights[1]:.4f}")
    print(f"w_lexical (Léxico):  {normalized_weights[2]:.4f}")
    print(f"w_semantic (Tono):   {normalized_weights[3]:.4f}")
    
    print("\nFórmula para el informe técnico:")
    print(f"H(s) = {normalized_weights[0]:.2f}·h_json + {normalized_weights[1]:.2f}·h_length + {normalized_weights[2]:.2f}·h_lexical + {normalized_weights[3]:.2f}·h_semantic")
    
    return normalized_weights

if __name__ == "__main__":
    optimize_heuristic_weights()