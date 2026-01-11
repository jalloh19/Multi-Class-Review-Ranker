# Multi-Class Review Ranker - Implementation Guide

## Project Structure

```
Multi-Class-Review-Ranker/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Text preprocessing with POS-lemmatization
│   ├── train.py           # Training all 3 models
│   ├── evaluate.py        # Model evaluation and comparison
│   ├── utils.py           # Utility functions
│   └── main.py            # Main entry point
├── data/
│   ├── raw/
│   │   └── Amazon_Reviews.csv
│   └── processed/
│       └── amazon_reviews_processed.csv
├── saved_models/
│   ├── naive_bayes_final.pkl
│   ├── svm_final.pkl
│   ├── word2vec_lr_final.pkl
│   └── word2vec.model
├── notebooks/
│   └── model.ipynb
└── app/
    └── app.py
```

## Step-by-Step Implementation

### Step 1: Data Preprocessing

**File: `src/preprocessing.py`**

Key functions:
- `preprocess_text(text, use_pos_tagging=True)` - Clean and lemmatize text
- `load_and_clean_data(file_path)` - Load and preprocess entire dataset
- `map_to_3_classes(rating)` - Convert 5-class to 3-class labels

**Usage:**
```python
from src.preprocessing import load_and_clean_data

# Load and clean data
df = load_and_clean_data('../data/raw/Amazon_Reviews.csv')
print(df.head())
```

**What it does:**
1. ✓ Loads raw Amazon reviews
2. ✓ Extracts numeric ratings (1-5)
3. ✓ Combines review title + text
4. ✓ Applies POS-aware lemmatization
5. ✓ Removes stopwords, URLs, HTML
6. ✓ Extracts additional features (word count, punctuation, etc.)
7. ✓ Removes duplicates and empty reviews

---

### Step 2: Train Model 1 - Naive Bayes

**File: `src/train.py`**

**Function:** `train_naive_bayes(X_train, y_train)`

**Usage:**
```python
from src.train import train_naive_bayes
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['rating'], test_size=0.2, random_state=42
)

# Train Naive Bayes
nb_model = train_naive_bayes(X_train, y_train)
```

**What it does:**
1. ✓ Creates TF-IDF + MultinomialNB pipeline
2. ✓ Performs GridSearchCV with 324 parameter combinations
3. ✓ Tunes: max_features, ngram_range, min_df, max_df, alpha
4. ✓ Uses 3-fold stratified cross-validation
5. ✓ Returns best estimator

**Best parameters found:**
- TF-IDF max_features: ~5000
- N-gram range: (1, 2) or (1, 3)
- Alpha: 0.1-1.0

---

### Step 3: Train Model 2 - SVM

**Function:** `train_svm(X_train, y_train)`

**Usage:**
```python
from src.train import train_svm

# Train SVM
svm_model = train_svm(X_train, y_train)
```

**What it does:**
1. ✓ Creates TF-IDF + LinearSVC pipeline
2. ✓ Performs GridSearchCV with 96 parameter combinations
3. ✓ Tunes: max_features, ngram_range, min_df, C, class_weight
4. ✓ Uses 3-fold stratified cross-validation
5. ✓ Returns best estimator

**Best parameters found:**
- TF-IDF max_features: ~5000-8000
- C: 0.5-2.0
- Class weight: balanced or None

---

### Step 4: Train Model 3 - Word2Vec

**Function:** `train_word2vec(X_train, y_train, X_test)`

**Usage:**
```python
from src.train import train_word2vec

# Train Word2Vec
w2v_model, lr_model, X_train_w2v, X_test_w2v = train_word2vec(
    X_train, y_train, X_test
)
```

**What it does:**
1. ✓ Tokenizes all reviews
2. ✓ Trains Word2Vec embeddings (100-dimensional vectors)
3. ✓ Creates document vectors (mean of word vectors)
4. ✓ Trains Logistic Regression on document vectors
5. ✓ Returns embeddings model, classifier, and vectors

**Parameters:**
- Vector size: 100
- Window: 5
- Min count: 2

---

### Step 5: Evaluate All Models

**File: `src/evaluate.py`**

**Function:** `evaluate_all_models(models_dict, X_test, y_test, X_test_w2v)`

**Usage:**
```python
from src.evaluate import evaluate_all_models

# Prepare models dictionary
models = {
    'naive_bayes': nb_model.best_estimator_,
    'svm': svm_model.best_estimator_,
    'word2vec_classifier': lr_model
}

# Evaluate
results = evaluate_all_models(models, X_test, y_test, X_test_w2v)
```

**What it does:**
1. ✓ Gets predictions from all 3 models
2. ✓ Calculates comprehensive metrics (accuracy, F1, precision, recall)
3. ✓ Generates comparison visualizations
4. ✓ Creates confusion matrices
5. ✓ Identifies best performing model
6. ✓ Returns results DataFrame

**Metrics calculated:**
- Accuracy
- Balanced Accuracy
- F1 Score (Weighted & Macro)
- Precision (Weighted)
- Recall (Weighted)

---

### Step 6: Complete Pipeline (All in One)

**Function:** `train_all_models(df)`

**Usage:**
```python
from src.train import train_all_models

# Train everything at once
results = train_all_models(
    df, 
    test_size=0.2, 
    use_3_class=True,
    models_dir='../saved_models'
)

# Access results
print(results['models'])  # All trained models
print(results['X_test'])  # Test data
print(results['y_test'])  # Test labels
```

**What it does:**
1. ✓ Splits data into train/test
2. ✓ Trains all 3 models sequentially
3. ✓ Saves all models to disk
4. ✓ Returns dictionary with models and data splits

---

## Command Line Usage

### Option 1: Using main.py

```bash
# Train all models
cd src
python main.py --mode train --data ../data/raw/Amazon_Reviews.csv --3class

# Evaluate existing models
python main.py --mode evaluate

# Interactive prediction
python main.py --mode predict
```

### Option 2: Python Script

Create `run_pipeline.py`:

```python
from src.preprocessing import load_and_clean_data
from src.train import train_all_models
from src.evaluate import evaluate_all_models, save_results

# 1. Load and preprocess data
print("Step 1: Loading data...")
df = load_and_clean_data('../data/raw/Amazon_Reviews.csv')

# 2. Train all models
print("Step 2: Training models...")
results = train_all_models(df, use_3_class=True)

# 3. Evaluate
print("Step 3: Evaluating...")
results_df = evaluate_all_models(
    results['models'],
    results['X_test'],
    results['y_test'],
    results['X_test_w2v']
)

# 4. Save results
save_results(results_df)

print("Done!")
```

Run:
```bash
python run_pipeline.py
```

---

## Making Predictions on New Reviews

```python
from src.utils import load_models, predict_sentiment
from src.preprocessing import preprocess_text

# Load models
models = load_models('../saved_models')

# New review
review = "This product is absolutely amazing! Best purchase ever!"

# Preprocess
clean_text = preprocess_text(review, use_pos_tagging=True)

# Predict with Naive Bayes
nb_pred = predict_sentiment(clean_text, models['naive_bayes'], 'pipeline')
print(f"Naive Bayes: {nb_pred}")

# Predict with SVM
svm_pred = predict_sentiment(clean_text, models['svm'], 'pipeline')
print(f"SVM: {svm_pred}")

# Predict with Word2Vec
w2v_pred = predict_sentiment(
    clean_text, 
    models['word2vec_classifier'], 
    'word2vec',
    models['word2vec_embeddings']
)
print(f"Word2Vec: {w2v_pred}")
```

---

## Expected Results

Based on notebook experiments:

| Model | Accuracy | F1 (Weighted) | Best Use Case |
|-------|----------|---------------|---------------|
| **SVM (Tuned)** | ~94-95% | ~0.94 | Best overall performance |
| **Naive Bayes (Tuned)** | ~91-92% | ~0.90 | Fastest training/prediction |
| **Word2Vec (POS-Lemma)** | ~87-89% | ~0.87 | Semantic understanding |

---

## Troubleshooting

### Issue: NLTK resources not found
```python
import nltk
nltk.download('all')
```

### Issue: Model files not found
Check that saved_models/ directory contains:
- naive_bayes_final.pkl
- svm_final.pkl
- word2vec_lr_final.pkl
- word2vec.model

### Issue: Memory error during training
Reduce parameter grid in train.py or use smaller dataset

---

## Next Steps

1. **Deploy best model** (SVM) to production
2. **Create Streamlit web application** in app/app.py
3. **Add model comparison interface**
4. **Deploy to Streamlit Cloud**
5. **Add user feedback collection**

---

## File Checklist

- [x] src/preprocessing.py - Complete
- [x] src/train.py - Complete with all 3 models
- [x] src/evaluate.py - Complete with visualizations
- [x] src/utils.py - Complete with helper functions
- [x] src/main.py - Complete CLI interface
- [x] This implementation guide

All models are ready to use!
