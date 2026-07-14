import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = {
    "score_json": ["C_format_json", "C_required_json_keys", "C_exact_lines"],
    "score_length": ["C_min_words", "C_max_words"],
    "score_lexical": ["C_mandatory_words", "C_forbidden_words"],
    "score_semantic": ["C_tone"],
}

DEFAULT_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])


def _default_path(*parts):
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), *parts)


def _constraint_series_to_float(series):
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "1": 1.0,
            "0": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.combine_first(numeric)


def _score_constraint_group(df, columns):
    available_columns = [column for column in columns if column in df.columns]
    if not available_columns:
        return pd.Series(1.0, index=df.index)

    values = pd.DataFrame(
        {column: _constraint_series_to_float(df[column]) for column in available_columns},
        index=df.index,
    )
    return values.mean(axis=1, skipna=True).fillna(1.0)


def build_training_frame(csv_path, strategy=None):
    df = pd.read_csv(csv_path)

    if strategy:
        strategies = {item.strip() for item in strategy.split(",") if item.strip()}
        df = df[df["generation_strategy"].isin(strategies)].copy()

    if df.empty:
        raise ValueError("No telemetry rows are available after applying the filters.")

    training_df = pd.DataFrame(index=df.index)
    for feature, columns in FEATURE_COLUMNS.items():
        training_df[feature] = _score_constraint_group(df, columns)

    if "global_score" not in df.columns:
        raise ValueError("Telemetry CSV must include a global_score column.")

    training_df["success"] = (pd.to_numeric(df["global_score"], errors="coerce") == 1.0).astype(int)

    if training_df["success"].nunique() < 2:
        raise ValueError("Training requires both successful and failing rows.")

    return training_df


def optimize_heuristic_weights(csv_path=None, output_path=None, strategy=None):
    print("--- Optimizador de Heuristicas (Machine Learning) ---")

    csv_path = csv_path or _default_path("data", "dynamic_experiment_results.csv")
    training_df = build_training_frame(csv_path, strategy=strategy)

    X = training_df[["score_json", "score_length", "score_lexical", "score_semantic"]]
    y = training_df["success"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)

    raw_weights = model.coef_[0]
    positive_weights = np.maximum(0, raw_weights)

    if np.sum(positive_weights) == 0:
        normalized_weights = DEFAULT_WEIGHTS
    else:
        normalized_weights = positive_weights / np.sum(positive_weights)

    result = {
        "w_json": float(normalized_weights[0]),
        "w_length": float(normalized_weights[1]),
        "w_lexical": float(normalized_weights[2]),
        "w_semantic": float(normalized_weights[3]),
        "source_csv": os.path.abspath(csv_path),
        "strategy_filter": strategy,
        "rows_used": int(len(training_df)),
        "success_rows": int(y.sum()),
        "failure_rows": int(len(y) - y.sum()),
    }

    print(f"\nFuente: {result['source_csv']}")
    print(
        f"Filas usadas: {result['rows_used']} "
        f"({result['success_rows']} exitosas, {result['failure_rows']} fallidas)"
    )
    print("\nEntrenamiento completado (con escalado de caracteristicas).")
    print("\n--- Nuevos Pesos Heuristicos Optimizados ---")
    print(f"w_json (Formato):    {result['w_json']:.4f}")
    print(f"w_length (Longitud): {result['w_length']:.4f}")
    print(f"w_lexical (Lexico):  {result['w_lexical']:.4f}")
    print(f"w_semantic (Tono):   {result['w_semantic']:.4f}")

    print("\nFormula para el informe tecnico:")
    print(
        "H(s) = "
        f"{result['w_json']:.2f}*h_json + "
        f"{result['w_length']:.2f}*h_length + "
        f"{result['w_lexical']:.2f}*h_lexical + "
        f"{result['w_semantic']:.2f}*h_semantic"
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"\nPesos guardados en: {os.path.abspath(output_path)}")

    return normalized_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Learn heuristic weights from experiment telemetry.")
    parser.add_argument("--csv", default=None, help="Path to dynamic_experiment_results.csv")
    parser.add_argument(
        "--output",
        default=_default_path("data", "heuristic_weights.json"),
        help="Path where the learned weights JSON will be written.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Optional comma-separated generation_strategy filter, for example: search,mcts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    optimize_heuristic_weights(csv_path=args.csv, output_path=args.output, strategy=args.strategy)
