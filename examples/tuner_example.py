"""
Example usage of the AutoTuner for automated hyperparameter optimization.

This script demonstrates how to use the AutoTuner with different configurations
and datasets for both classification and regression tasks.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Import the AutoTuner
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tune.tuner import AutoTuner, AutoTunerConfig


def create_sample_classification_data():
    """Create a sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def create_sample_regression_data():
    """Create a sample regression dataset."""
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        noise=0.1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def example_basic_classification():
    """Basic classification example with XGBoost."""
    print("="*60)
    print("EXAMPLE 1: Basic Classification with XGBoost")
    print("="*60)
    
    # Create sample data
    df = create_sample_classification_data()
    
    # Create tuner with minimal configuration
    tuner = AutoTuner(
        dataset=df,
        target="target",
        estimator_type="xgboost",
        task_type="classification",
        metric="accuracy"
    )
    
    # Run tuning with fewer iterations for demo
    tuner.tune(max_iterations=2)
    
    # Get results
    # best_config = tuner.get_best_config()
    # summary = tuner.get_tuning_summary()
    
    # print(f"\nBest configuration: {best_config}")
    # print(f"Best score: {summary['best_score']:.4f}")


def example_custom_config_regression():
    """Regression example with custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Regression with Custom Configuration")
    print("="*60)
    
    # Create sample data
    df = create_sample_regression_data()
    
    # Create custom configuration
    config = AutoTunerConfig(
        n_trials=50,  # Fewer trials for demo
        max_iterations=3,
        cv_folds=3,
        n_repeats=1,
        show_progress_bar=True,
        save_results=False  # Don't save for demo
    )
    
    # Create tuner
    tuner = AutoTuner(
        config=config,
        dataset=df,
        target="target",
        estimator_type="lightgbm",
        task_type="regression",
        metric="r2"
    )
    
    # Run tuning
    tuner.tune()
    
    # Get results
    # best_configs = tuner.get_best_configs(n=3)
    # print("\nTop 3 configurations:")
    # for i, config in enumerate(best_configs, 1):
    #     print(f"{i}. Score: {config['score']:.4f}, Config: {config['config']}")


def example_custom_estimator():
    """Example with custom sklearn estimator."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Estimator (Random Forest)")
    print("="*60)
    
    # Create sample data
    df = create_sample_classification_data()
    
    # Create custom estimator
    custom_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Create tuner with custom estimator
    tuner = AutoTuner(
        dataset=df,
        target="target",
        estimator=custom_rf,
        estimator_type="random_forest",  # This helps with search space generation
        task_type="classification",
        metric="f1"
    )
    
    # Run tuning with minimal iterations
    tuner.tune(max_iterations=2)
    
    # Get results
    # summary = tuner.get_tuning_summary()
    # print(f"\nTuning completed in {summary['iterations']} iterations")
    # print(f"Best score: {summary['best_score']:.4f}")
    # print(f"Score progression: {summary['progression']}")


def example_from_yaml_config():
    """Example using YAML configuration file."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Using YAML Configuration")
    print("="*60)
    
    # Create sample data
    df = create_sample_regression_data()
    
    # Load configuration from YAML file
    config_path = "src/tune/config.yml"
    
    try:
        # Create tuner with YAML config
        tuner = AutoTuner(
            config_path=config_path,
            dataset=df,
            target="target",
            estimator_type="xgboost",
            task_type="regression",
            metric="neg_mean_squared_error"
        )
        
        # Override some config values for demo
        tuner.config.max_iterations = 2
        tuner.config.n_trials = 30
        tuner.config.save_results = False
        
        # Run tuning
        tuner.tune(user_description="Predicting continuous target values from synthetic features")
        
        print("\nTuning completed successfully with YAML configuration!")
        
    except Exception as e:
        print(f"Error loading YAML config: {e}")
        print("Make sure the config.yml file exists in src/tune/")


def main():
    """Run all examples."""
    print("🚀 AutoTuner Examples")
    print("This script demonstrates various ways to use the AutoTuner")
    
    try:
        # Run examples
        example_basic_classification()
        example_custom_config_regression()
        example_custom_estimator()
        example_from_yaml_config()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
