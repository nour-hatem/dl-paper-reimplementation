import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
VAL_SIZE = 0.10
TEST_SIZE = 0.10
LABEL = "Sentiment_label"
LABELS = ["Negative", "Neutral", "Positive"]


ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / "data" / "processed" / "arsas_clean.csv"
SPLITS_DIR = ROOT / "data" / "splits"
WEIGHTS_PATH = ROOT / "configs" / "class_weights.json"


def main():
    df = pd.read_csv(CLEAN_PATH, encoding="utf-8-sig")
    assert set(df[LABEL]) == set(LABELS), f"Unexpected labels: {set(df[LABEL])}"


    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[LABEL],
        random_state=SEED,
    )


    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val[LABEL],
        random_state=SEED,
    )

    splits = {"train": train, "val": val, "test": test}

    assert sum(len(s) for s in splits.values()) == len(df), "Rows lost or added"
    for column in ["#Tweet_ID", "Tweet_text"]:
        combined = pd.concat([s[column] for s in splits.values()])
        assert combined.is_unique, f"Repeated {column} in the splits"

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(SPLITS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    print("Seed:", SEED)
    print("\nRows per split:")
    for name, split in splits.items():
        print(f"  {name:<5} {len(split):>6}  ({len(split) / len(df):.1%})")

    distribution = pd.DataFrame(
        {name: s[LABEL].value_counts(normalize=True) for name, s in splits.items()}
    )
    print("\nLabel distribution (%):")
    print(distribution.mul(100).round(2))
    # 5. Class weights for the loss function, from the training set only
    weights = compute_class_weight(
        class_weight="balanced", classes=np.array(LABELS), y=train[LABEL]
    )
    class_weights = {label: float(w) for label, w in zip(LABELS, weights)}

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_PATH.write_text(json.dumps(class_weights, indent=2), encoding="utf-8")

    print("\nClass weights:")
    for label, weight in class_weights.items():
        print(f"  {label:<9} {weight:.4f}")

    print("\nAll checks passed.")
    print("Splits saved to:", SPLITS_DIR)
    print("Weights saved to:", WEIGHTS_PATH)


if __name__ == "__main__":
    main()