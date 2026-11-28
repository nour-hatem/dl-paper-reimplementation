from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "arsas_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "arsas_clean.csv"
CONFLICTS_PATH = ROOT / "results" / "label_conflicts.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["Tweet_text"] = df["Tweet_text"].str.strip()

key = (
    df["Tweet_text"]
    .str.replace(r"https?://\S+|@\w+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
key = key.mask(key.eq(""), df["Tweet_text"])

id_conflict = df.groupby("#Tweet_ID")["Sentiment_label"].transform("nunique") > 1
text_conflict = df.groupby(key)["Sentiment_label"].transform("nunique") > 1
conflict = id_conflict | text_conflict


clean = df[~conflict].drop_duplicates(subset="#Tweet_ID")
clean = clean[~key.loc[clean.index].duplicated()]
clean = clean.drop(columns="__index_level_0__").reset_index(drop=True)

print("Raw rows:", len(df))
print("Removed (conflicting labels):", conflict.sum())
print("Removed (duplicates):", len(df) - conflict.sum() - len(clean))
print("Rows with ID conflict:", id_conflict.sum())
print("Rows with text conflict:", text_conflict.sum())
print("Rows with both:", (id_conflict & text_conflict).sum())
print(clean["Sentiment_label"].value_counts())

assert clean["#Tweet_ID"].is_unique, "Repeated IDs remain"
assert not clean["Tweet_text"].duplicated().any(), "Repeated texts remain"
assert clean[["Tweet_text", "Sentiment_label"]].notna().all().all(), "Missing values"
assert clean["Tweet_text"].ne("").all(), "Empty tweets"
print("All checks passed. Clean rows:", len(clean))

CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

CONFLICTS_PATH.parent.mkdir(parents=True, exist_ok=True)
df.assign(id_conflict=id_conflict, text_conflict=text_conflict)[conflict] \
  .sort_values(["#Tweet_ID", "Tweet_text"]) \
  .to_csv(CONFLICTS_PATH, index=False, encoding="utf-8-sig")

print("Saved:", CLEAN_PATH)
print("Saved:", CONFLICTS_PATH)