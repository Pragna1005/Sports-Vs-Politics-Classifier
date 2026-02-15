# Sports-Vs-Politics-Classifier

This project implements a machine learning system that classifies news articles into **Sports** or **Politics** categories.

The goal of this project is to compare multiple feature representation techniques and machine learning models to determine which combination performs best for text classification.

---

## 📌 Project Overview

Text classification is a core task in Natural Language Understanding (NLU). In this project:

- News articles are classified as **SPORTS** or **POLITICS**
- Three feature engineering techniques are compared:
  - Bag of Words (BoW)
  - TF-IDF
  - TF-IDF with Bigrams
- Three machine learning models are evaluated:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)

---

## Dataset Description

The dataset (`df_file.csv`) originally contained five categories.  
For this project, only two were selected:

- **0 → POLITICS**
- **1 → SPORTS**

### Final Dataset Size:

- **Politics:** 417 articles  
- **Sports:** 511 articles  
- **Total:** 928 articles  

The dataset contains full news articles, making the classification realistic and meaningful.

---

## Preprocessing

The following preprocessing steps were applied:

- Converted text to lowercase
- Removed punctuation and numerical characters
- Removed special symbols

No aggressive stemming or lemmatization was applied to preserve contextual meaning.

---

## Feature Engineering Techniques

### Bag of Words (BoW)
Represents documents as raw word frequency vectors.

### TF-IDF
Weights words based on importance across documents.

### TF-IDF + Bigrams
Captures both single words and two-word phrases such as:
- "prime minister"
- "world cup"

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Linear Support Vector Machine (SVM)**

---

## 📊 Model Performance Comparison

| Feature Method | Naive Bayes | Logistic Regression | SVM |
|---------------|------------|--------------------|------|
| Bag of Words | 1.00 | 0.97 | 0.98 |
| TF-IDF | 1.00 | 0.98 | 1.00 |
| TF-IDF + Bigrams | 1.00 | 0.99 | 1.00 |

### Observations

- TF-IDF performs better than simple Bag of Words.
- Including bigrams improves contextual understanding.
- SVM achieves the highest overall performance.
- The dataset appears highly separable, leading to near-perfect classification. In real-world scenarios with more nuanced or overlapping content, performance may be lower.

---
