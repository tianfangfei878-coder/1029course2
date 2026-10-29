import csv
from pathlib import Path

import jieba
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Hiragino Sans GB", "DejaVu Sans"]

ROOT = Path(__file__).resolve().parent.parent
NOVEL_PATH = ROOT / "corpora" / "牵风记-徐怀中.txt"
MODEL_PATH = ROOT / "outputs" / "sentiment_model.joblib"
CSV_PATH = ROOT / "outputs" / "sentiment_arc.csv"
PNG_PATH = ROOT / "outputs" / "sentiment_arc.png"

TARGET_CHARS = 500
WINDOW = 10
SENTENCE_ENDS = "。！？"


def split_passages(text, target=TARGET_CHARS):
    sentences = []
    buf = []
    for ch in text:
        buf.append(ch)
        if ch in SENTENCE_ENDS:
            sentences.append("".join(buf))
            buf = []
    if buf:
        sentences.append("".join(buf))

    passages = []
    current, length = [], 0
    for s in sentences:
        current.append(s)
        length += len(s)
        if length >= target:
            passages.append("".join(current))
            current, length = [], 0
    if current:
        passages.append("".join(current))
    return passages


def main():
    print(f"Loading model from {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    vectorizer, model = bundle["vectorizer"], bundle["model"]

    text = NOVEL_PATH.read_text(encoding="utf-8")
    print(f"Novel: {len(text)} characters")

    passages = split_passages(text)
    print(f"Passages: {len(passages)}")

    scores = []
    for i, passage in enumerate(passages, 1):
        seg = " ".join(jieba.lcut(passage))
        X = vectorizer.transform([seg])
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        p_pos = proba[classes.index(1)]
        p_neg = proba[classes.index(0)]
        scores.append(p_pos - p_neg)

    df = pd.DataFrame({
        "passage_number": range(1, len(passages) + 1),
        "score": scores,
        "text": passages,
    })
    df.to_csv(CSV_PATH, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    print(f"Saved scores to {CSV_PATH}")

    moving_avg = df["score"].rolling(WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["passage_number"], df["score"], color="lightgray",
            label="Passage score")
    ax.plot(df["passage_number"], moving_avg, color="tab:blue", linewidth=2,
            label=f"{WINDOW}-passage moving average")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Passage number")
    ax.set_ylabel("Score  (P(positive) - P(negative))")
    ax.set_title("Sentiment arc of 牵风记 (徐怀中)")
    ax.legend(loc="lower left")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {PNG_PATH}")

    print("\n3 highest-scoring passages:")
    for _, row in df.nlargest(3, "score").iterrows():
        print(f"  #{int(row.passage_number)}  score={row.score:+.4f}  {row.text[:80]}...")
    print("\n3 lowest-scoring passages:")
    for _, row in df.nsmallest(3, "score").iterrows():
        print(f"  #{int(row.passage_number)}  score={row.score:+.4f}  {row.text[:80]}...")


if __name__ == "__main__":
    main()
