"""
Quick start example for Multi-Class Review Ranker
This script demonstrates the complete workflow.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_and_clean_data
from src.train import train_all_models
from src.evaluate import evaluate_all_models, save_results

def main():
    print("="*70)
    print("MULTI-CLASS REVIEW RANKER - QUICK START")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n📁 STEP 1: Loading and preprocessing data...")
    print("-" * 70)
    data_path = '../data/raw/Amazon_Reviews.csv'
    
    try:
        df = load_and_clean_data(data_path, use_pos_tagging=True)
        print(f"✓ Successfully loaded {len(df)} reviews")
        print(f"✓ Features: {list(df.columns)}")
    except FileNotFoundError:
        print(f"✗ Error: Could not find {data_path}")
        print("  Please ensure the data file exists in the correct location.")
        return
    
    # Step 2: Train all three models
    print("\n🎯 STEP 2: Training all three models...")
    print("-" * 70)
    print("This will train:")
    print("  1. Naive Bayes (with hyperparameter tuning)")
    print("  2. SVM (with hyperparameter tuning)")
    print("  3. Word2Vec + Logistic Regression")
    print("\nThis may take 10-15 minutes depending on your hardware...")
    
    results = train_all_models(
        df, 
        test_size=0.2, 
        use_3_class=True,  # Use 3-class: Negative, Neutral, Positive
        models_dir='../saved_models'
    )
    
    print("\n✓ All models trained successfully!")
    
    # Step 3: Evaluate and compare models
    print("\n📊 STEP 3: Evaluating and comparing models...")
    print("-" * 70)
    
    results_df = evaluate_all_models(
        results['models'],
        results['X_test'],
        results['y_test'],
        results['X_test_w2v'],
        show_plots=False  # Set to True to see visualizations
    )
    
    # Save results
    save_results(results_df, '../data/processed/model_comparison.csv')
    
    # Step 4: Summary
    print("\n📈 FINAL SUMMARY")
    print("="*70)
    print("\nModel Performance Ranking:")
    print(results_df[['Model', 'Accuracy', 'F1 (Weighted)']].to_string(index=False))
    
    best_model = results_df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Accuracy: {best_model['Accuracy']:.2%}")
    print(f"   F1 Score: {best_model['F1 (Weighted)']:.4f}")
    
    print("\n✅ COMPLETE! Models saved to: saved_models/")
    print("   - naive_bayes_final.pkl")
    print("   - svm_final.pkl")
    print("   - word2vec_lr_final.pkl")
    print("   - word2vec.model")
    
    print("\n💡 Next steps:")
    print("   1. Run 'python src/main.py --mode predict' for interactive predictions")
    print("   2. Check IMPLEMENTATION_GUIDE.md for detailed usage")
    print("   3. Deploy the best model to production")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
