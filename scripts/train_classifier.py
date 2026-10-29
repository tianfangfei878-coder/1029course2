import json
import logging
from pathlib import Path

import jieba
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "corpora" / "douban.json"
MODEL_PATH = ROOT / "outputs" / "sentiment_model.joblib"

TRAIN_PER_CLASS = 5000
TEST_PER_CLASS = 1000
RANDOM_STATE = 42

jieba.setLogLevel(logging.WARNING)


def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    positives = [d["text"] for d in data if d["label"] == 5]  # 5-star
    negatives = [d["text"] for d in data if d["label"] == 1]  # 1-star
    return positives, negatives


def split(positive, negative, rng):
    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for label, texts in ((1, positive), (0, negative)):
        order = rng.permutation(len(texts))
        train_idx = order[:TRAIN_PER_CLASS]
        test_idx = order[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]
        train_texts.extend(texts[i] for i in train_idx)
        train_labels.extend([label] * TRAIN_PER_CLASS)
        test_texts.extend(texts[i] for i in test_idx)
        test_labels.extend([label] * TEST_PER_CLASS)
    return train_texts, train_labels, test_texts, test_labels


def main():
    print(f"Loading {DATA_PATH}")
    positive, negative = load_data(DATA_PATH)
    print(f"5-star reviews: {len(positive)}, 1-star reviews: {len(negative)}")

    rng = np.random.default_rng(RANDOM_STATE)
    train_texts, train_labels, test_texts, test_labels = split(positive, negative, rng)
    print(f"Training on {len(train_texts)} reviews, testing on {len(test_texts)} reviews")

    print("Segmenting with jieba...")
    train_seg = [" ".join(jieba.lcut(t)) for t in train_texts]
    test_seg = [" ".join(jieba.lcut(t)) for t in test_texts]

    print("Fitting TF-IDF vectorizer on training reviews...")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_seg)
    X_test = vectorizer.transform(test_seg)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("Training LogisticRegression...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, train_labels)

    accuracy = model.score(X_test, test_labels)
    print(f"\nTest accuracy ({len(test_texts)} reviews): {accuracy:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, MODEL_PATH)
    print(f"Saved vectorizer and model to {MODEL_PATH}")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]

    top_positive_idx = np.argsort(coefs)[-20:][::-1]
    print("\nTop 20 most positive words:")
    for i in top_positive_idx:
        print(f"  {feature_names[i]}\t{coefs[i]:.4f}")

    top_negative_idx = np.argsort(coefs)[:20]
    print("\nTop 20 most negative words:")
    for i in top_negative_idx:
        print(f"  {feature_names[i]}\t{coefs[i]:.4f}")


if __name__ == "__main__":
    main()
