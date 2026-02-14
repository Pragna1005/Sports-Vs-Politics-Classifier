import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# -------------------------
# Load Dataset
# -------------------------
def load_data():
    df = pd.read_csv("../data/df_file.csv")

    # Keep only labels 0 and 1
    df = df[df["Label"].isin([0, 1])]

    # Convert labels to names
    df["Label"] = df["Label"].map({0: "POLITICS", 1: "SPORTS"})

    # Clean text
    df["Text"] = df["Text"].apply(clean_text)

    return df


# -------------------------
# Text Cleaning
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


# -------------------------
# Evaluate Function
# -------------------------
def evaluate_models(X_train, X_test, y_train, y_test, title):
    print(f"\n===== {title} =====")

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")


# -------------------------
# Main
# -------------------------
def main():

    df = load_data()

    X = df["Text"]
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1️⃣ Bag of Words
    bow = CountVectorizer()
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)
    evaluate_models(X_train_bow, X_test_bow, y_train, y_test, "Bag of Words")

    # 2️⃣ TF-IDF
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF")

    # 3️⃣ TF-IDF + Bigrams
    tfidf_bigram = TfidfVectorizer(ngram_range=(1,2))
    X_train_bigram = tfidf_bigram.fit_transform(X_train)
    X_test_bigram = tfidf_bigram.transform(X_test)
    evaluate_models(X_train_bigram, X_test_bigram, y_train, y_test, "TF-IDF + Bigrams")


if __name__ == "__main__":
    main()

