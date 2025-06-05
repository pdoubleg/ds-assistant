"""
Example usage of the GeneralizedTuner class.

This script demonstrates how to use the GeneralizedTuner with different estimators
and workflows: dataset-aware and user-description-based tuning.
"""

import sys
import os
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score
from sklearn.model_selection import train_test_split

from src.modules.tune import AutoTuneLLM


def create_sample_dataset() -> tuple[pd.DataFrame, str]:
    """
    Create a sample binary classification dataset for demonstration.
    
    Returns:
        tuple[pd.DataFrame, str]: Dataset and target column name
    """
    # Generate a synthetic binary classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, 'target'


def example_xgboost_with_dataset_analysis():
    """
    Example: XGBoost tuning with dataset analysis workflow.
    """
    print("=" * 60)
    print("EXAMPLE 1: XGBoost with Dataset Analysis")
    print("=" * 60)
    
    # Create sample dataset
    df, target = create_sample_dataset()
    
    # Initialize tuner for XGBoost with dataset analysis
    tuner = AutoTuneLLM(
        estimator_type="xgboost",
        dataset=df,
        target=target,
        scoring=make_scorer(precision_score, pos_label=1),
        task_description="Binary classification task with balanced classes. Goal is to maximize precision.",
        top_n_configs=3,
        max_consecutive_no_improvement=2
    )
    
    # Run tuning with dataset analysis
    tuner.tune_with_dataset_analysis(max_iterations=3)
    
    # Get results
    best_config = tuner.get_best_config()
    summary = tuner.get_tuning_summary()
    
    print(f"\nBest Configuration: {best_config}")
    print(f"Best Score: {summary['best_score']:.4f}")
    print(f"Total Iterations: {summary['total_iterations']}")
    
    # Create and show the best estimator
    best_estimator = tuner.create_best_estimator()
    print(f"Best Estimator: {type(best_estimator).__name__}")


def example_lightgbm_with_user_description():
    """
    Example: LightGBM tuning with user description workflow.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: LightGBM with User Description")
    print("=" * 60)
    
    # Create sample dataset
    df, target = create_sample_dataset()
    
    # Define detailed user description
    user_description = """
    This is a binary classification problem with the following characteristics:
    - Dataset contains 1000 samples with 10 features
    - The target variable is binary (0/1) 
    - Classes are reasonably balanced
    - Features are numerical and have been preprocessed
    - The goal is to maximize F1-score as we care about both precision and recall
    - We expect some features to be more informative than others
    - The problem domain suggests that tree-based models should work well
    - We want to avoid overfitting given the moderate sample size
    """
    
    # Initialize tuner for LightGBM with user description workflow
    tuner = AutoTuneLLM(
        estimator_type="lightgbm",
        dataset=df,  # Still needed for actual tuning, just not passed to LLM
        target=target,
        scoring=make_scorer(f1_score),
        top_n_configs=3,
        max_consecutive_no_improvement=2
    )
    
    # Run tuning with user description (no dataset analysis passed to LLM)
    tuner.tune_with_user_description(user_description, max_iterations=3)
    
    # Get results
    best_config = tuner.get_best_config()
    summary = tuner.get_tuning_summary()
    
    print(f"\nBest Configuration: {best_config}")
    print(f"Best Score: {summary['best_score']:.4f}")
    print(f"Total Iterations: {summary['total_iterations']}")


def example_custom_estimator():
    """
    Example: Custom estimator (Random Forest) tuning with dataset analysis.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Estimator (Random Forest)")
    print("=" * 60)
    
    # Create sample dataset
    df, target = create_sample_dataset()
    
    # Create a custom Random Forest estimator
    custom_rf = RandomForestClassifier(random_state=42)
    
    # Initialize tuner for custom estimator
    tuner = AutoTuneLLM(
        estimator_type="custom",
        custom_estimator=custom_rf,
        dataset=df,
        target=target,
        scoring="accuracy",  # Use string scoring
        task_description="Binary classification with Random Forest. Focus on generalization and avoid overfitting.",
        top_n_configs=3,
        max_consecutive_no_improvement=2
    )
    
    # Run tuning with dataset analysis
    tuner.tune_with_dataset_analysis(max_iterations=2)
    
    # Get results
    best_config = tuner.get_best_config()
    summary = tuner.get_tuning_summary()
    
    print(f"\nBest Configuration: {best_config}")
    print(f"Best Score: {summary['best_score']:.4f}")
    print(f"Total Iterations: {summary['total_iterations']}")


def example_comparison():
    """
    Example: Compare multiple tuning approaches on the same dataset.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Comparison of Different Approaches")
    print("=" * 60)
    
    # Create sample dataset
    df, target = create_sample_dataset()
    
    # Split into train/test for final evaluation
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    
    results = {}
    
    # Test XGBoost
    print("\nTuning XGBoost...")
    xgb_tuner = AutoTuneLLM(
        estimator_type="xgboost",
        dataset=train_df,
        target=target,
        scoring=make_scorer(f1_score),
        max_consecutive_no_improvement=1
    )
    xgb_tuner.tune_with_dataset_analysis(max_iterations=2)
    results['XGBoost'] = xgb_tuner.get_tuning_summary()
    
    # Test LightGBM  
    print("\nTuning LightGBM...")
    lgb_tuner = AutoTuneLLM(
        estimator_type="lightgbm",
        dataset=train_df,
        target=target, 
        scoring=make_scorer(f1_score),
        max_consecutive_no_improvement=1
    )
    
    user_desc = "Binary classification task with moderate dataset size. Optimize for F1-score."
    lgb_tuner.tune_with_user_description(user_desc, max_iterations=2)
    results['LightGBM'] = lgb_tuner.get_tuning_summary()
    
    # Compare results
    print("\n" + "=" * 40)
    print("COMPARISON RESULTS")
    print("=" * 40)
    
    for model_name, summary in results.items():
        print(f"{model_name}:")
        print(f"  Best Score: {summary['best_score']:.4f}")
        print(f"  Iterations: {summary['total_iterations']}")
        print(f"  Improvement: {summary.get('improvement_over_baseline', 'N/A')}")
        print()


if __name__ == "__main__":
    """
    Run all examples to demonstrate the GeneralizedTuner functionality.
    
    Note: These examples use small iteration counts for demonstration purposes.
    In practice, you would typically use more iterations for better results.
    """
    print("GeneralizedTuner Examples")
    print("=" * 60)
    print("This script demonstrates the flexibility of the GeneralizedTuner class.")
    print("Note: Using small iteration counts for quick demonstration.\n")
    
    try:
        # Run individual examples
        example_xgboost_with_dataset_analysis()
        example_lightgbm_with_user_description() 
        example_custom_estimator()
        example_comparison()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have all required dependencies installed.")
        print("You may need to set up your OpenAI API key as well.") 