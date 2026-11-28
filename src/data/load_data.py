from datasets import load_dataset

dataset = load_dataset(
    "Qanadil/ArSAS_An_Arabic_Speech-Act_and_Sentiment_Corpus_of_Tweets"
)

print(dataset)
print(dataset["train"][0])


df = dataset["train"].to_pandas()

df.to_csv(
    "arsas_raw.csv",
    index=False,
    encoding="utf-8-sig"
)