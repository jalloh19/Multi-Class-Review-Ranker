"""
Utility functions for the Multi-Class Review Ranker project.
"""

import os
import sys
import joblib
import logging
from gensim.models import Word2Vec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_project_root():
    """
    Return the absolute path of the project root.
    
    Returns:
        String path to project root
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models(models_dir='../saved_models'):
    """
    Load all trained models from disk.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        Dictionary containing all loaded models
    """
    models = {}
    
    logger.info(f"Loading models from {models_dir}...")
    try:
        models['naive_bayes'] = joblib.load(f'{models_dir}/naive_bayes_final.pkl')
        logger.info("✓ Naive Bayes loaded")
    except FileNotFoundError:
        logger.warning("✗ Naive Bayes model not found")
    
    try:
        models['svm'] = joblib.load(f'{models_dir}/svm_final.pkl')
        logger.info("✓ SVM loaded")
    except FileNotFoundError:
        logger.warning("✗ SVM model not found")
    
    try:
        models['word2vec_classifier'] = joblib.load(f'{models_dir}/word2vec_lr_final.pkl')
        logger.info("✓ Word2Vec Classifier loaded")
    except FileNotFoundError:
        logger.warning("✗ Word2Vec Classifier not found")
    
    try:
        models['word2vec_embeddings'] = Word2Vec.load(f'{models_dir}/word2vec.model')
        logger.info("✓ Word2Vec Embeddings loaded")
    except FileNotFoundError:
        logger.warning("✗ Word2Vec Embeddings not found")
    
    return models


def predict_sentiment(text, model, model_type='pipeline', w2v_model=None):
    """
    Predict sentiment for a single review text.
    
    Args:
        text: Review text to classify
        model: Trained model
        model_type: Type of model ('pipeline' for NB/SVM, 'word2vec' for Word2Vec)
        w2v_model: Word2Vec embeddings model (required if model_type='word2vec')
        
    Returns:
        Predicted class label
    """
    if model_type == 'pipeline':
        # For Naive Bayes and SVM (sklearn pipelines)
        return model.predict([text])[0]
    
    elif model_type == 'word2vec':
        # For Word2Vec + Logistic Regression
        import numpy as np
        
        if w2v_model is None:
            raise ValueError("w2v_model is required for word2vec model_type")
        
        # Tokenize
        tokens = text.split()
        
        # Create document vector
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if len(vectors) == 0:
            doc_vector = np.zeros(w2v_model.vector_size)
        else:
            doc_vector = np.mean(vectors, axis=0)
        
        # Predict
        return model.predict([doc_vector])[0]
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def setup_paths():
    """
    Setup and verify project directory structure.
    
    Returns:
        Dictionary of important paths
    """
    root = get_project_root()
    
    paths = {
        'root': root,
        'data': os.path.join(root, 'data'),
        'raw_data': os.path.join(root, 'data', 'raw'),
        'processed_data': os.path.join(root, 'data', 'processed'),
        'models': os.path.join(root, 'saved_models'),
        'notebooks': os.path.join(root, 'notebooks'),
        'src': os.path.join(root, 'src')
    }
    
    # Verify paths exist
    for name, path in paths.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} (not found)")
    
    return paths


def print_model_info(models_dict):
    """
    Print information about loaded models.
    
    Args:
        models_dict: Dictionary of loaded models
    """
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    if 'naive_bayes' in models_dict:
        print("\n1. Naive Bayes (Tuned)")
        nb = models_dict['naive_bayes']
        print(f"   Type: {type(nb).__name__}")
        if hasattr(nb, 'named_steps'):
            print(f"   Pipeline steps: {list(nb.named_steps.keys())}")
    
    if 'svm' in models_dict:
        print("\n2. SVM (Tuned)")
        svm = models_dict['svm']
        print(f"   Type: {type(svm).__name__}")
        if hasattr(svm, 'named_steps'):
            print(f"   Pipeline steps: {list(svm.named_steps.keys())}")
    
    if 'word2vec_embeddings' in models_dict and 'word2vec_classifier' in models_dict:
        print("\n3. Word2Vec + Logistic Regression")
        w2v = models_dict['word2vec_embeddings']
        lr = models_dict['word2vec_classifier']
        print(f"   Embeddings type: {type(w2v).__name__}")
        print(f"   Classifier type: {type(lr).__name__}")
        print(f"   Vocabulary size: {len(w2v.wv)}")
        print(f"   Vector dimension: {w2v.vector_size}")
