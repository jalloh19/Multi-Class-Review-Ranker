# Source Code (src/) - Implementation Overview

This directory contains the complete implementation of the Multi-Class Review Ranker using three different models.

## 📁 Files

### 1. `preprocessing.py`
**Text preprocessing and data loading**

Key functions:
- `preprocess_text(text, use_pos_tagging=True)` - Advanced text cleaning with POS-aware lemmatization
- `load_and_clean_data(file_path)` - Load raw data and apply all preprocessing steps
- `map_to_3_classes(rating)` - Convert 5-star ratings to 3-class labels

Features:
- ✓ POS-tagging for context-aware lemmatization
- ✓ Stopword removal
- ✓ URL and HTML cleaning
- ✓ English word filtering
- ✓ Feature extraction (word count, punctuation, etc.)

### 2. `train.py`
**Model training with hyperparameter tuning**

Implements three models:

**Model 1: Naive Bayes**
- `train_naive_bayes(X_train, y_train)`
- TF-IDF + MultinomialNB pipeline
- GridSearchCV with 324 parameter combinations
- Best parameters: max_features=5000, ngram_range=(1,2), alpha=0.1

**Model 2: SVM**
- `train_svm(X_train, y_train)`
- TF-IDF + LinearSVC pipeline
- GridSearchCV with 96 parameter combinations
- Best parameters: max_features=8000, C=1.0, class_weight='balanced'

**Model 3: Word2Vec**
- `train_word2vec(X_train, y_train, X_test)`
- Custom Word2Vec embeddings (100-dim)
- Logistic Regression classifier
- Semantic understanding of reviews

**All-in-one:**
- `train_all_models(df)` - Train all three models and save them

### 3. `evaluate.py`
**Model evaluation and comparison**

Key functions:
- `evaluate_model(model, X_test, y_test)` - Evaluate single model
- `evaluate_all_models(models_dict, X_test, y_test)` - Compare all models
- `plot_confusion_matrix(y_true, y_pred, title)` - Visualize confusion matrix
- `plot_metrics_comparison(results_df)` - Compare model metrics
- `analyze_errors(y_true, y_pred, X_test)` - Error analysis

Metrics:
- Accuracy
- Balanced Accuracy
- F1 Score (Weighted & Macro)
- Precision
- Recall

### 4. `utils.py`
**Utility functions**

- `load_models(models_dir)` - Load all saved models
- `predict_sentiment(text, model, model_type)` - Predict on single review
- `setup_paths()` - Verify project structure
- `print_model_info(models_dict)` - Display model information

### 5. `main.py`
**Command-line interface**

Three modes:

```bash
# Train all models
python main.py --mode train --3class

# Evaluate existing models
python main.py --mode evaluate

# Interactive prediction
python main.py --mode predict
```

## 🚀 Quick Usage

### Option 1: Use the complete pipeline

```python
from src.preprocessing import load_and_clean_data
from src.train import train_all_models

# Load data
df = load_and_clean_data('../data/raw/Amazon_Reviews.csv')

# Train everything
results = train_all_models(df, use_3_class=True)

# Models are automatically saved to ../saved_models/
```

### Option 2: Train models individually

```python
from src.preprocessing import load_and_clean_data
from src.train import train_naive_bayes, train_svm, train_word2vec
from sklearn.model_selection import train_test_split

# Load and split data
df = load_and_clean_data('../data/raw/Amazon_Reviews.csv')
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['rating'], test_size=0.2, random_state=42
)

# Train Naive Bayes
nb_model = train_naive_bayes(X_train, y_train)

# Train SVM
svm_model = train_svm(X_train, y_train)

# Train Word2Vec
w2v_model, lr_model, X_train_w2v, X_test_w2v = train_word2vec(
    X_train, y_train, X_test
)
```

### Option 3: Evaluate existing models

```python
from src.utils import load_models
from src.evaluate import evaluate_all_models

# Load saved models
models = load_models('../saved_models')

# Evaluate
results = evaluate_all_models(
    models, X_test, y_test, X_test_w2v, show_plots=True
)
```

### Option 4: Make predictions

```python
from src.utils import load_models, predict_sentiment
from src.preprocessing import preprocess_text

# Load models
models = load_models('../saved_models')

# Preprocess new review
review = "This product is amazing!"
clean_review = preprocess_text(review, use_pos_tagging=True)

# Predict with each model
nb_pred = predict_sentiment(clean_review, models['naive_bayes'], 'pipeline')
svm_pred = predict_sentiment(clean_review, models['svm'], 'pipeline')
w2v_pred = predict_sentiment(
    clean_review, 
    models['word2vec_classifier'], 
    'word2vec',
    models['word2vec_embeddings']
)

print(f"Predictions: NB={nb_pred}, SVM={svm_pred}, W2V={w2v_pred}")
```

## 📊 Expected Performance

| Model | Accuracy | Training Time | Prediction Speed |
|-------|----------|---------------|------------------|
| Naive Bayes | 91-92% | ~5 min | Very Fast |
| SVM | 94-95% | ~10 min | Fast |
| Word2Vec | 87-89% | ~8 min | Medium |

**Recommendation:** Use SVM for best accuracy, Naive Bayes for fastest predictions.

## 🔧 Dependencies

```python
pandas
numpy
scikit-learn
gensim
nltk
matplotlib
seaborn
tqdm
joblib
```

Install all:
```bash
pip install -r ../requirements.txt
```

## 📝 Notes

- All models use **3-class classification** by default (Negative, Neutral, Positive)
- Can switch to 5-class by setting `use_3_class=False`
- POS-tagging improves Word2Vec accuracy by ~2-3%
- Models are saved in pickle/binary format for fast loading
- GridSearchCV uses 3-fold stratified cross-validation
- All random states set to 42 for reproducibility

## 🎯 Next Steps

1. Test with `quick_start.py`
2. Read `IMPLEMENTATION_GUIDE.md` for detailed steps
3. Deploy best model (SVM) to production
4. Create Streamlit interface in `app/app.py`
