import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from learn_heuristics import build_training_frame


def test_build_training_frame_uses_telemetry_csv(tmp_path):
    csv_path = tmp_path / "telemetry.csv"
    pd.DataFrame(
        [
            {
                "generation_strategy": "mcts",
                "global_score": 1.0,
                "C_format_json": True,
                "C_required_json_keys": True,
                "C_min_words": True,
                "C_max_words": True,
                "C_mandatory_words": True,
                "C_forbidden_words": True,
                "C_tone": True,
            },
            {
                "generation_strategy": "mcts",
                "global_score": 0.5,
                "C_format_json": False,
                "C_required_json_keys": True,
                "C_min_words": False,
                "C_max_words": True,
                "C_mandatory_words": False,
                "C_forbidden_words": True,
                "C_tone": False,
            },
        ]
    ).to_csv(csv_path, index=False)

    training_df = build_training_frame(csv_path)

    assert training_df["score_json"].tolist() == [1.0, 0.5]
    assert training_df["score_length"].tolist() == [1.0, 0.5]
    assert training_df["score_lexical"].tolist() == [1.0, 0.5]
    assert training_df["score_semantic"].tolist() == [1.0, 0.0]
    assert training_df["success"].tolist() == [1, 0]
