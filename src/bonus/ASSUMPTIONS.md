# Assumptions and decisions

This file records every assumption or implementation choice that is **not** explicitly stated in Wadhawan (2021), *AraBERT and Farasa Segmentation Based Approach For Sarcasm and Sentiment Detection in Arabic Tweets* (arXiv 2103.01679), EACL WANLP-2021 Shared Task 2, Subtask 2. Each entry is also reflected in a code comment or docstring near the relevant implementation.

## Phase A — Exploratory data analysis

- **Raw schema.** The raw files were inspected before writing code. They contain exactly `tweet`, `sarcasm`, `sentiment`, and `dialect`. `sarcasm` is parsed by pandas as a boolean, `sentiment` and `dialect` as strings, and `tweet` as a string.
- **Test labels are real.** The test file contains a populated `sentiment` column with only `POS`, `NEG`, and `NEU`. These labels are treated as held-out gold. They are never used for training, hyperparameter tuning, or model selection; only the separate final evaluation function may read them.
- **Duplicate and overlap matching.** Duplicate counts and the primary train/test overlap check use exact string equality on the raw tweet text. A separate normalized overlap check strips leading/trailing whitespace and collapses internal whitespace runs. Neither check applies Unicode normalisation or lowercasing.
- **Character length.** Tweet length in characters is the number of Unicode code points (`len` on the Python string), not the number of grapheme clusters.
- **Whitespace-token length.** Whitespace-token length is the number of tokens produced by Python's `str.split()` with no arguments.
- **AraBERT token length.** Token lengths come from `aubmindlab/bert-base-arabertv02` with `add_special_tokens=True`, no truncation, and no padding. The reported length therefore includes `[CLS]` and `[SEP]`.
- **Paper comparison.** The paper reports 12,548 training rows, about 17% positive examples, and a 90:10 split of 11,293 training / 1,255 validation rows. The EDA compares the observed row count and positive share directly; the split sizes are compared in Phase B.
- **EDA does not modify data.** No rows are dropped, deduplicated, or relabelled during Phase A.

## Phase A review decisions

- **Paper title.** The title is recorded as *AraBERT and Farasa Segmentation Based Approach For Sarcasm and Sentiment Detection in Arabic Tweets* (Wadhawan, arXiv 2103.01679) in the notebook and in this file.
- **Plot palette.** All EDA figures use shades of blue only (`BLUE_DARK`, `BLUE_MEDIUM`, `BLUE_LIGHT`).
- **Neutral label check.** `check_sentiment_labels` returns a split-independent verdict (`VALID SENTIMENT LABELS` or `EMPTY OR PLACEHOLDER LABELS`). The held-out-gold warning is printed only in the notebook cell that checks the test split.
- **Conflicting duplicate labels.** Conflicting sentiment labels are detected only for tweets whose exact text appears more than once, consistent with the exact duplicate counts. No whitespace normalisation is used for this check. In the training file this finds 25 repeated tweets covering 50 rows.
- **Normalized overlap.** The normalized overlap strips leading/trailing whitespace and collapses internal whitespace runs, but does not lowercase, remove punctuation, or apply Unicode normalisation. For this dataset it finds the same 2 overlapping tweets as the exact check.
- **Marker definitions.** URLs match `https?://\S+|www\.\S+`, mentions match `(?<!\w)@\w+`, and hashtags match `(?<!\w)#\w+`. `\w` is Unicode-aware, so Arabic letters are included.
- **Marker ratio.** Tokens-per-word ratios in the marker breakdown are aggregate ratios (total AraBERT tokens divided by total whitespace tokens for the relevant subset), not the mean of per-tweet ratios. AraBERT token counts include special tokens.
- **Label shift.** The train/test sentiment label shift is reported in the notebook summary but is not corrected in EDA; no resampling, reweighting, or relabelling is applied.
