"""
CLI Demo for AutoTuner

This script creates sample data and demonstrates how to use the AutoTuner
from the command line interface.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os
import subprocess
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_demo_datasets():
    """Create demo datasets for CLI testing."""
    print("Creating demo datasets...")
    
    # Classification dataset
    X, y = make_classification(
        n_samples=500,  # Smaller for demo
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df_class = pd.DataFrame(X, columns=feature_names)
    df_class['target'] = y
    df_class.to_csv("demo_classification.csv", index=False)
    print("✅ Created demo_classification.csv")
    
    # Regression dataset
    X, y = make_regression(
        n_samples=500,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df_reg = pd.DataFrame(X, columns=feature_names)
    df_reg['target'] = y
    df_reg.to_csv("demo_regression.csv", index=False)
    print("✅ Created demo_regression.csv")


def run_cli_examples():
    """Run CLI examples."""
    print("\n" + "="*60)
    print("CLI EXAMPLES")
    print("="*60)
    
    # Example 1: Basic classification
    print("\n1. Basic Classification Example:")
    print("Command: python src/tune/tuner.py --data demo_classification.csv --target target --estimator xgboost --metric accuracy --max-iterations 2 --n-trials 20 --no-save")
    
    # Example 2: Regression with config
    print("\n2. Regression with Custom Config:")
    print("Command: python src/tune/tuner.py --config src/tune/config.yml --data demo_regression.csv --target target --estimator lightgbm --metric r2 --max-iterations 2 --no-save")
    
    # Example 3: Interactive mode
    print("\n3. Interactive Mode:")
    print("Command: python src/tune/tuner.py --interactive")
    
    # Example 4: Random Forest classification
    print("\n4. Random Forest Classification:")
    print("Command: python src/tune/tuner.py --data demo_classification.csv --target target --estimator random_forest --metric f1 --max-iterations 2 --n-trials 15 --no-save")
    
    print("\n" + "="*60)
    print("To run these examples, copy and paste the commands above")
    print("Make sure you're in the project root directory")
    print("="*60)


def cleanup_demo_files():
    """Clean up demo files."""
    files_to_remove = ["demo_classification.csv", "demo_regression.csv"]
    
    print("\nCleaning up demo files...")
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✅ Removed {file}")


def main():
    """Main function."""
    print("🎯 AutoTuner CLI Demo")
    print("This script creates sample datasets and shows CLI usage examples")
    
    try:
        # Create demo datasets
        create_demo_datasets()
        
        # Show CLI examples
        run_cli_examples()
        
        # Ask user if they want to clean up
        response = input("\nDo you want to keep the demo CSV files? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            cleanup_demo_files()
        else:
            print("Demo files kept. You can use them to test the CLI commands above.")
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
