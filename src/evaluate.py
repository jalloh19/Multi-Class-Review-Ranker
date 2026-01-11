"""
Evaluation module for all three models.
Provides comprehensive metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

sns.set_style('darkgrid')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_metrics(name, y_true, y_pred):
    """
    Calculate comprehensive metrics for a model.
    
    Args:
        name: Model name
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1 (Weighted)': f1_score(y_true, y_pred, average='weighted'),
        'F1 (Macro)': f1_score(y_true, y_pred, average='macro'),
        'Precision (Weighted)': precision_score(y_true, y_pred, average='weighted'),
        'Recall (Weighted)': recall_score(y_true, y_pred, average='weighted')
    }


def plot_confusion_matrix(y_true, y_pred, title, figsize=(12, 6)):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    
    sns.set_style('darkgrid')
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(results_df, figsize=(12, 6)):
    """
    Plot comparison of key metrics across models.
    
    Args:
        results_df: DataFrame with metrics for all models
        figsize: Figure size
    """
    sns.set_style('darkgrid')
    plt.figure(figsize=figsize)
    
    long_results = results_df.melt(
        id_vars='Model', 
        value_vars=['Accuracy', 'F1 (Weighted)'], 
        var_name='Metric', 
        value_name='Score'
    )
    
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=long_results, palette='viridis')
    plt.title('Model Comparison: Key Metrics', fontsize=16)
    plt.ylim(0.5, 1.0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Add value labels to bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, model_name, X_test_vectors=None):
    """
    Evaluate a single model and return predictions.
    
    Args:
        model: Trained model
        X_test: Test data
        y_test: True labels
        model_name: Name of the model
        X_test_vectors: Pre-computed vectors (for Word2Vec)
        
    Returns:
        Predictions
    """
    print(f"\nEvaluating {model_name}...")
    
    if X_test_vectors is not None:
        y_pred = model.predict(X_test_vectors)
    else:
        y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred


def evaluate_all_models(models_dict, X_test, y_test, X_test_w2v=None, show_plots=True):
    """
    Evaluate all three models and compare results.
    
    Args:
        models_dict: Dictionary containing all trained models
        X_test: Test text data
        y_test: True labels
        X_test_w2v: Word2Vec test vectors
        show_plots: Whether to show visualizations
        
    Returns:
        DataFrame with results for all models
    """
    print("\n" + "="*60)
    logger.info("="*60)
    logger.info("EVALUATING ALL MODELS")
    logger.info("="*60)
    
    # Get predictions
    logger.info("Evaluating Naive Bayes...")
    y_pred_nb = evaluate_model(
        models_dict['naive_bayes'], X_test, y_test, "Naive Bayes (Tuned)"
    )
    
    logger.info("Evaluating SVM...")
    y_pred_svm = evaluate_model(
        models_dict['svm'], X_test, y_test, "SVM (Tuned)"
    )
    
    logger.info("Evaluating Word2Vec...")
    y_pred_w2v = evaluate_model(
        models_dict['word2vec_classifier'], None, y_test, 
        "Word2Vec (POS-Lemma)", X_test_vectors=X_test_w2v
    )
    
    # Calculate metrics
    metrics_data = [
        get_metrics('Naive Bayes (Tuned)', y_test, y_pred_nb),
        get_metrics('SVM (Tuned)', y_test, y_pred_svm),
        get_metrics('Word2Vec (POS-Lemma)', y_test, y_pred_w2v)
    ]
    
    results_df = pd.DataFrame(metrics_data).sort_values('Accuracy', ascending=False)
    
    logger.info("="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"\n{results_df.round(4).to_string(index=False)}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(results_df.round(4).to_string(index=False))
    
    # Identify best model
    best_model = results_df.iloc[0]
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model['Model']}")
    print(f"Accuracy: {best_model['Accuracy']:.2%}")
    print(f"{'='*60}")
    
    if show_plots:
        # Plot metrics comparison
        plot_metrics_comparison(results_df)
        
        # Plot confusion matrices
        plot_confusion_matrix(y_test, y_pred_nb, 'Naive Bayes (Tuned) - Confusion Matrix')
        plot_confusion_matrix(y_test, y_pred_svm, 'SVM (Tuned) - Confusion Matrix')
        plot_confusion_matrix(y_test, y_pred_w2v, 'Word2Vec (POS-Lemma) - Confusion Matrix')
    
    return results_df


def analyze_errors(y_true, y_pred, X_test, model_name, n_samples=5):
    """
    Analyze misclassified samples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        X_test: Test data
        model_name: Name of the model
        n_samples: Number of error samples to show
    """
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS - {model_name}")
    print(f"{'='*60}")
    
    error_indices = np.where(y_pred != y_true)[0]
    print(f"Total Misclassified: {len(error_indices)} out of {len(y_true)}")
    print(f"Error Rate: {len(error_indices)/len(y_true):.2%}")
    
    if len(error_indices) > 0:
        np.random.seed(99)
        sample_errors = np.random.choice(error_indices, min(n_samples, len(error_indices)), replace=False)
        
        error_records = []
        for i in sample_errors:
            text_val = X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i]
            true_val = y_true.iloc[i] if hasattr(y_true, 'iloc') else y_true[i]
            pred_val = y_pred[i]
            
            error_records.append({
                'Review Text': text_val[:100] + '...',
                'True Label': true_val,
                'Predicted': pred_val
            })
        
        error_df = pd.DataFrame(error_records)
        print("\nSample Errors:")
        print(error_df.to_string(index=False))


def save_results(results_df, output_path='../data/processed/model_comparison.csv'):
    """
    Save evaluation results to CSV.
    
    Args:
        results_df: DataFrame with model results
        output_path: Path to save CSV
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

