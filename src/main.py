"""
Main script for training and evaluating all three models.

Usage:
    python main.py --mode train    # Train all models
    python main.py --mode evaluate # Evaluate existing models
    python main.py --mode predict  # Make predictions on new text
"""

import argparse
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_and_clean_data
from src.train import train_all_models
from src.evaluate import evaluate_all_models, analyze_errors, save_results
from src.utils import load_models, predict_sentiment, setup_paths, print_model_info


def train_pipeline(data_path, use_3_class=True, models_dir='../saved_models'):
    """
    Complete training pipeline for all three models.
    
    Args:
        data_path: Path to raw data CSV
        use_3_class: Use 3-class or 5-class classification
        models_dir: Directory to save models
    """
    print("\n" + "="*60)
    print("STARTING TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\nStep 1/3: Loading and preprocessing data...")
    df = load_and_clean_data(data_path, use_pos_tagging=True)
    print(f"✓ Loaded {len(df)} samples")
    
    # Step 2: Train all models
    print("\nStep 2/3: Training models...")
    results = train_all_models(df, test_size=0.2, use_3_class=use_3_class, models_dir=models_dir)
    
    # Step 3: Evaluate models
    print("\nStep 3/3: Evaluating models...")
    results_df = evaluate_all_models(
        results['models'],
        results['X_test'],
        results['y_test'],
        results['X_test_w2v'],
        show_plots=False
    )
    
    # Save results
    save_results(results_df)
    
    logger.info("="*60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*60)
    
    return results_df


def evaluate_pipeline(models_dir='../saved_models', data_path='../data/processed/amazon_reviews_processed.csv'):
    """
    Evaluate existing trained models.
    
    Args:
        models_dir: Directory containing saved models
        data_path: Path to processed data
    """
    try:
        logger.info("="*60)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info("="*60)
        
        # Load models
        logger.info("Loading models...")
        models = load_models(models_dir)
        print_model_info(models)
        
        # Load data
        logger.info("Loading test data...")
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from src.preprocessing import map_to_3_classes
        import numpy as np
        
        df = pd.read_csv(data_path)
        
        # Prepare data
        X_text = df['text']
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure models are trained first (run with --mode train)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {e}")
        raise
    y = df['rating'].apply(map_to_3_classes)
    
    # Split (using same seed as training)
    _, X_test, _, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create Word2Vec vectors
    X_test_tokens = [text.split() for text in X_test.astype(str)]
    w2v_model = models['word2vec_embeddings']
    
    def get_mean_vector(word_list, model):
        vectors = [model.wv[word] for word in word_list if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)
    
    X_test_w2v = np.array([get_mean_vector(tokens, w2v_model) for tokens in X_test_tokens])
    
    # Evaluate
    results_df = evaluate_all_models(models, X_test, y_test, X_test_w2v, show_plots=True)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


def predict_pipeline(models_dir='../saved_models'):
    """
    Interactive prediction mode.
    
    Args:
        models_dir: Directory containing saved models
    """
    try:
        logger.info("="*60)
        logger.info("INTERACTIVE PREDICTION MODE")
        logger.info("="*60)
        
        # Load models
        logger.info("Loading models...")
        models = load_models(models_dir)
        
        if not models:
            logger.error("No models found. Train models first with --mode train")
            sys.exit(1)
        
        # Preprocess function
        from src.preprocessing import preprocess_text
        
        print("\nEnter review text to classify (or 'quit' to exit):")
        
        while True:
            text = input("\nReview: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Preprocess
            clean_text = preprocess_text(text, use_pos_tagging=True)
            
            print(f"\nCleaned text: {clean_text}\n")
            
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise
        # Predict with each model
        try:
            nb_pred = predict_sentiment(clean_text, models['naive_bayes'], 'pipeline')
            print(f"Naive Bayes prediction: {nb_pred}")
        except:
            print("Naive Bayes: Error")
        
        try:
            svm_pred = predict_sentiment(clean_text, models['svm'], 'pipeline')
            print(f"SVM prediction: {svm_pred}")
        except:
            print("SVM: Error")
        
        try:
            w2v_pred = predict_sentiment(
                clean_text, 
                models['word2vec_classifier'], 
                'word2vec',
                models['word2vec_embeddings']
            )
            print(f"Word2Vec prediction: {w2v_pred}")
        except Exception as e:
            logger.error(f"Word2Vec error: {e}")
            print(f"Word2Vec: Error - {e}")


def main():
    """Main entry point."""
    try:
        parser = argparse.ArgumentParser(description='Multi-Class Review Ranker')
        parser.add_argument(
            '--mode',
            type=str,
            choices=['train', 'evaluate', 'predict'],
            default='train',
            help='Operation mode: train, evaluate, or predict'
        )
        parser.add_argument(
            '--data',
            type=str,
            default='../data/raw/Amazon_Reviews.csv',
            help='Path to data file'
        )
        parser.add_argument(
            '--models-dir',
            type=str,
            default='../saved_models',
            help='Directory for saved models'
        )
        parser.add_argument(
            '--3class',
            action='store_true',
            help='Use 3-class classification instead of 5-class'
        )
        
        args = parser.parse_args()
        
        # Setup paths
        logger.info("Verifying project structure...")
        setup_paths()
        
        # Execute requested mode
        if args.mode == 'train':
            train_pipeline(args.data, use_3_class=args.3class or True, models_dir=args.models_dir)
        
        elif args.mode == 'evaluate':
            evaluate_pipeline(models_dir=args.models_dir)
    
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    
    elif args.mode == 'predict':
        predict_pipeline(models_dir=args.models_dir)


if __name__ == '__main__':
    main()
