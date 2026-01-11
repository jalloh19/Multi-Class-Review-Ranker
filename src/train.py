"""
Training module for all three models:
1. Naive Bayes (Tuned)
2. SVM (Tuned)
3. Word2Vec with Logistic Regression
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_naive_bayes(X_train, y_train, cv_folds=3, verbose=1):
    """
    Train and tune Naive Bayes classifier using GridSearchCV.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        verbose: Verbosity level
        
    Returns:
        Trained GridSearchCV object with best estimator
    """
    print("\n" + "="*60)
    print("TRAINING NAIVE BAYES")
    print("="*60)
    
    # Create pipeline
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    
    # Parameter grid
    param_grid = {
        'tfidf__max_features': [3000, 5000, 8000],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2, 3],
        'tfidf__max_df': [0.9, 0.95, 1.0],
        'clf__alpha': [0.01, 0.1, 0.5, 1.0]
    }
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    logger.info(f"Training Naive Bayes with {3*3*3*3*4} parameter combinations")
    logger.info("This may take a few minutes...")
    
    # Grid search
    grid_search = GridSearchCV(
        nb_pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=verbose
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best CV Score: {grid_search.best_score_:.4f}")
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    
    print("\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV Accuracy: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    return grid_search


def train_svm(X_train, y_train, cv_folds=3, verbose=1):
    """
    Train and tune SVM classifier using GridSearchCV.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        cv_folds: Number of cross-validation folds
        verbose: Verbosity level
        
    Returns:
        Trained GridSearchCV object with best estimator
    """
    print("\n" + "="*60)
    print("TRAINING SVM")
    print("="*60)
    
    # Create pipeline
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC(max_iter=2000))
    ])
    
    # Parameter grid
    param_grid = {
        'tfidf__max_features': [3000, 5000, 8000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [1, 2],
        'clf__C': [0.1, 0.5, 1.0, 2.0],
        'clf__class_weight': [None, 'balanced']
    }
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    logger.info(f"Training SVM with {3*2*2*4*2} parameter combinations")
    logger.info("This may take several minutes...")
    
    # Grid search
    grid_search = GridSearchCV(
        svm_pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=verbose
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best CV Score: {grid_search.best_score_:.4f}")
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    
    print("\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV Accuracy: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    return grid_search


def train_word2vec(X_train, y_train, X_test, vector_size=100, window=5, min_count=2):
    """
    Train Word2Vec embeddings and Logistic Regression classifier.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        X_test: Test text data (for vocabulary building)
        vector_size: Dimension of word vectors
        window: Context window size
        min_count: Minimum word frequency
        
    Returns:
        Tuple of (Word2Vec model, trained Logistic Regression, train vectors, test vectors)
    """
    print("\n" + "="*60)
    print("TRAINING WORD2VEC + LOGISTIC REGRESSION")
    print("="*60)
    
    # Tokenize
    def tokenize_corpus(text_series):
        return [text.split() for text in text_series.astype(str)]
    
    X_train_tokens = tokenize_corpus(X_train)
    X_test_tokens = tokenize_corpus(X_test)
    
    # Combine for training embeddings
    full_corpus = X_train_tokens + X_test_tokens
    
    logger.info(f"Training Word2Vec on {len(full_corpus)} documents...")
    
    # Train Word2Vec
    w2v_model = Word2Vec(
        sentences=full_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42
    )
    
    logger.info(f"Vocabulary size: {len(w2v_model.wv)}")
    
    # Create document vectors
    def get_mean_vector(word_list, model):
        vectors = [model.wv[word] for word in word_list if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)
    
    print("Creating document vectors...")
    X_train_w2v = np.array([get_mean_vector(tokens, w2v_model) for tokens in X_train_tokens])
    X_test_w2v = np.array([get_mean_vector(tokens, w2v_model) for tokens in X_test_tokens])
    
    print(f"Train vectors shape: {X_train_w2v.shape}")
    print(f"Test vectors shape: {X_test_w2v.shape}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression classifier...")
    lr_model = LogisticRegression(max_iter=2000, random_state=42)
    lr_model.fit(X_train_w2v, y_train)
    
    print("Training complete!")
    
    return w2v_model, lr_model, X_train_w2v, X_test_w2v


def train_all_models(df, test_size=0.2, use_3_class=True, models_dir='../saved_models'):
    """
    Train all three models on the dataset.
    
    Args:
        df: Preprocessed DataFrame
        test_size: Test set proportion
        use_3_class: Whether to use 3-class (True) or 5-class (False) classification
        models_dir: Directory to save models
        
    Returns:
        Dictionary containing all trained models and data splits
    """
    logger.info("="*60)
    logger.info("TRAINING ALL MODELS")
    logger.info("="*60)
    
    # Prepare data
    X_text = df['text']
    y = df['rating']
    
    # Map to 3 classes if needed
    if use_3_class:
        from src.preprocessing import map_to_3_classes
        y = y.apply(map_to_3_classes)
        logger.info("Using 3-class classification (Negative, Neutral, Positive)")
    else:
        logger.info("Using 5-class classification")
    
    logger.info(f"Class distribution:\n{y.value_counts().sort_index()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    models = {}
    
    # 1. Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    models['naive_bayes'] = nb_model.best_estimator_
    
    # 2. SVM
    svm_model = train_svm(X_train, y_train)
    models['svm'] = svm_model.best_estimator_
    
    # 3. Word2Vec
    w2v_model, lr_model, X_train_w2v, X_test_w2v = train_word2vec(X_train, y_train, X_test)
    models['word2vec_embeddings'] = w2v_model
    models['word2vec_classifier'] = lr_model
    
    # Save models
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    joblib.dump(models['naive_bayes'], f'{models_dir}/naive_bayes_final.pkl')
    print(f"✓ Naive Bayes saved to: {models_dir}/naive_bayes_final.pkl")
    
    joblib.dump(models['svm'], f'{models_dir}/svm_final.pkl')
    print(f"✓ SVM saved to: {models_dir}/svm_final.pkl")
    
    joblib.dump(models['word2vec_classifier'], f'{models_dir}/word2vec_lr_final.pkl')
    w2v_model.save(f'{models_dir}/word2vec.model')
    print(f"✓ Word2Vec Classifier saved to: {models_dir}/word2vec_lr_final.pkl")
    print(f"✓ Word2Vec Embeddings saved to: {models_dir}/word2vec.model")
    
    return {
        'models': models,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_w2v': X_train_w2v,
        'X_test_w2v': X_test_w2v
    }

