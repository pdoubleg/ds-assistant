"""
Example: MLflow AutoTuner Integration

This example demonstrates how to use the MLflowAutoTuner for hyperparameter optimization
with natural MLflow integration. It shows both the full API and convenience functions.

Usage:
    python examples/mlflow_autotuner_example.py
"""
import os
import sys
import warnings
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tune.tune_mlflow import MLflowAutoTuner, run_mlflow_tuning

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


os.environ["LOKY_MAX_CPU_COUNT"] = '4'
# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ['PYTHONWARNINGS']='ignore'


def create_sample_classification_data():
    """Create a sample classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def create_sample_regression_data():
    """Create a sample regression dataset."""
    X, y = make_regression(
        n_samples=1000, n_features=15, n_informative=10, noise=0.1, random_state=42
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    return df


def example_full_api():
    """Example using the full MLflowAutoTuner API."""
    print("=" * 60)
    print("Example 1: Full MLflowAutoTuner API")
    print("=" * 60)

    # Create sample data
    df = create_sample_classification_data()

    # Initialize the tuner
    tuner = MLflowAutoTuner(
        dataset=df,
        config_path="src/tune/config.yml",
        metric="roc_auc",
        experiment_name="autotuner_example_full_api",
    )
    tuner.config.task_type = "classification"
    tuner.config.target_column = "target"
    tuner.deps.target_column = "target"
    # Run tuning
    results = tuner.tune(max_iterations=3)

    # Get results
    # best_config = tuner.get_best_config()
    # summary = tuner.get_tuning_summary()

    # print(f"\nBest Configuration: {best_config}")
    # print(f"Best Score: {summary['best_score']:.4f}")
    # print(f"Total Iterations: {summary['iterations']}")
    # print(f"Result usage: {results['usage']}")

    return results


# def example_convenience_function():
#     """Example using the convenience function."""
#     print("\n" + "=" * 60)
#     print("Example 2: Convenience Function")
#     print("=" * 60)

#     # Create sample data
#     df = create_sample_regression_data()

#     # Run tuning with convenience function
#     results = run_mlflow_tuning(
#         dataset=df,
#         config_path="src/tune/config.yml",
#         metric="r2",
#         experiment_name="autotuner_example_convenience"
#     )
#     # print(f"\nBest Configuration: {results['best_config']}")
#     # print(f"Best Score: {results['best_score']:.4f}")
#     # print(f"Improvement over Baseline: {results['improvement']:.4f}")
#     # print(f"Result usage: {results['usage']}")

#     return results


def example_with_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    # Create sample data
    df = create_sample_classification_data()

    # Create custom estimator
    custom_lgbm = LGBMClassifier(random_state=42, verbose=-1)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lgbm", custom_lgbm),
        ]
    )

    # Initialize tuner with simplified interface
    # Configuration is loaded from the config file
    tuner = MLflowAutoTuner(
        dataset=df,
        config_path="src/tune/config.yml",
        metric="roc_auc",
        experiment_name="autotuner_example_custom_config",
        custom_estimator=pipeline,
    )
    tuner.config.task_type = "classification"
    tuner.config.metric = "roc_auc"
    tuner.deps.target_column = "target"
    tuner.config.target_column = "target"
    tuner.config.estimator_type = "lightgbm"

    # Run tuning
    results = tuner.tune(
        user_description="Binary classification task for customer churn prediction"
    )
    return results


def example_integration_with_existing_pipeline():
    """Example showing integration with existing MLflow pipeline."""
    print("\n" + "=" * 60)
    print("Example 4: Integration with Existing Pipeline")
    print("=" * 60)

    try:
        import mlflow

        # Set experiment
        mlflow.set_experiment("autotuner_pipeline_integration")

        with mlflow.start_run(run_name="main_pipeline"):
            # Log some pipeline metadata
            mlflow.log_param("pipeline_version", "1.0")
            mlflow.log_param("data_source", "synthetic")

            # Create sample data
            df = create_sample_classification_data()

            # Run AutoTuner as part of the pipeline
            with mlflow.start_run(nested=True, run_name="hyperparameter_optimization"):
                tuner = MLflowAutoTuner(
                    dataset=df,
                    config_path="src/tune/config.yml",
                    metric="accuracy",
                )
                tuner.config.task_type = "classification"

                tuner.config.target_column = "target"

                results = tuner.tune(max_iterations=2)

                # Log final results to parent run
                mlflow.log_metric("final_best_score", results["best_score"])
                mlflow.log_param("best_hyperparameters", str(results["best_config"]))

            # Continue with rest of pipeline...
            mlflow.log_metric("pipeline_completion", 1.0)

            print(f"\nPipeline completed with best score: {results['best_score']:.4f}")

    except ImportError:
        print("MLflow not available, skipping MLflow integration example")

        # Still run the tuning without MLflow
        df = create_sample_classification_data()
        results = run_mlflow_tuning(dataset=df, metric="accuracy")
        tuner.config.task_type = "classification"
        tuner.deps.target_column = "target"
        # print(f"Tuning completed without MLflow. Best score: {results['best_score']:.4f}")
        # print(f"Result usage: {results['usage']}")


def example_with_lightgbm_regression():
    """Example using LightGBM for regression with the penguins dataset.
    
    This example demonstrates how to use the MLflowAutoTuner with LightGBM
    for a regression task using the famous penguins dataset, including 
    comprehensive preprocessing and feature engineering steps.
    
    Returns:
        dict: Results from the tuning process including best configuration,
              scores, and other metadata.
    """
    print("\n" + "=" * 60)
    print("Example 6: LightGBM Regression with Penguins Dataset") 
    print("=" * 60)

    # Load the penguins dataset
    import seaborn as sns
    penguins_df = sns.load_dataset('penguins')
    
    # Drop rows with missing values for simplicity
    penguins_df = penguins_df.dropna()
    
    print(penguins_df.head())
    
    # Use body_mass_g as target for regression
    # Remove it from features and make it the target
    target_col = 'body_mass_g'
    
    print(f"Dataset shape: {penguins_df.shape}")
    print(f"Target variable: {target_col}")
    print(f"Features: {[col for col in penguins_df.columns if col != target_col]}")
        
    # Create comprehensive preprocessing and feature engineering pipeline
    
    # Identify categorical and numerical columns
    categorical_features = ['species', 'island', 'sex']
    numerical_features = [col for col in penguins_df.columns 
                         if col not in categorical_features + [target_col]]
    
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            # Numerical features: scaling and polynomial features
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
            ]), numerical_features),
            
            # Categorical features: one-hot encoding
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough',  # Keep any remaining columns
        verbose_feature_names_out=True  # This helps preserve feature names
    )
    
    # Create LightGBM regressor with reasonable defaults
    lgbm_regressor = LGBMRegressor(
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=-1,  # Suppress LightGBM warnings
        force_col_wise=True,  # Better for small datasets
    )
    
    # Create full pipeline with feature engineering and selection
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Handle categorical and numerical features
        ('feature_selection', SelectKBest(score_func=f_regression, k=10)),  # Tunable feature selection
        ('regressor', lgbm_regressor)  # LightGBM with tunable hyperparameters
    ])

    # Initialize tuner with custom configuration
    tuner = MLflowAutoTuner(
        dataset=penguins_df,
        config_path="src/tune/config.yml", 
        metric="r2",
        experiment_name="autotuner_example_lightgbm_penguins",
        custom_estimator=pipeline,
    )

    # Configure for regression task
    tuner.config.task_type = "regression"
    tuner.config.metric = "r2"
    tuner.deps.target_column = target_col
    tuner.config.target_column = target_col
    tuner.config.estimator_type = "lightgbm"
    tuner.deps.estimator_type = "lightgbm"
    tuner.deps.task_type = "regression"
    tuner.config.direction = "maximize"
    tuner.deps.direction = "maximize"
    
    # Set reasonable optimization parameters
    tuner.config.n_trials = 50  # More trials for better optimization
    tuner.config.cv_folds = 5   # 5-fold cross-validation
    tuner.config.verbose = 2    # Show detailed progress

    # Run tuning with descriptive task description
    results = tuner.tune(
        max_iterations=3,  # Allow multiple refinement iterations
        user_description=(
            "Regression task to predict penguin body mass using LightGBM. "
            "The dataset contains penguin measurements including bill dimensions, "
            "flipper length, species, island, and sex. The pipeline includes "
            "comprehensive preprocessing with polynomial feature engineering, "
            "feature selection, and dimensionality reduction before LightGBM modeling."
        )
    )
    
    return results


def example_with_lightgbm_classification():
    """Example using LightGBM for classification with the penguins dataset.
    
    This example demonstrates how to use the MLflowAutoTuner with LightGBM
    for a multi-class classification task to predict penguin species using 
    the famous penguins dataset, including comprehensive preprocessing and 
    feature engineering steps.
    
    Returns:
        dict: Results from the tuning process including best configuration,
              scores, and other metadata.
    """
    print("\n" + "=" * 60)
    print("Example 7: LightGBM Classification - Penguin Species Prediction") 
    print("=" * 60)

    # Load the penguins dataset
    import seaborn as sns
    penguins_df = sns.load_dataset('penguins')
    
    # Drop rows with missing values for simplicity
    penguins_df = penguins_df.dropna()
    
    # Use species as target for classification
    target_col = 'species'
           
    # Create comprehensive preprocessing and feature engineering pipeline
    
    # Identify categorical and numerical columns (excluding target)
    categorical_features = ['island', 'sex']  # species is now the target
    numerical_features = [col for col in penguins_df.columns 
                         if col not in categorical_features + [target_col]]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            # Numerical features: scaling and polynomial features
            ('num', Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
            ]), numerical_features),
            
            # Categorical features: one-hot encoding
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough',  # Keep any remaining columns
        verbose_feature_names_out=True  # This helps preserve feature names
    )
    
    # Create LightGBM classifier with reasonable defaults for multi-class
    lgbm_classifier = LGBMClassifier(
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=-1,  # Suppress LightGBM warnings
        force_col_wise=True,  # Better for small datasets
        objective='multiclass',  # Explicitly set for multi-class
        num_class=3,  # Three penguin species
    )
    
    # Create full pipeline with feature engineering and selection
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Handle categorical and numerical features
        ('feature_selection', SelectKBest(score_func=f_classif, k=15)),  # Tunable feature selection for classification
        ('classifier', lgbm_classifier)  # LightGBM with tunable hyperparameters
    ])
    # Initialize tuner with custom configuration
    tuner = MLflowAutoTuner(
        dataset=penguins_df,
        config_path="src/tune/config.yml", 
        metric="accuracy",
        experiment_name="autotuner_example_lightgbm_penguins_classification",
        custom_estimator=pipeline,
    )

    # Configure for classification task
    tuner.config.task_type = "classification"
    tuner.config.metric = "accuracy"
    tuner.deps.target_column = target_col
    tuner.config.target_column = target_col
    tuner.config.estimator_type = "lightgbm"
    tuner.deps.estimator_type = "lightgbm"
    tuner.deps.task_type = "classification"
    tuner.config.direction = "maximize"  # Maximize accuracy
    tuner.deps.direction = "maximize"
    
    tuner.deps.metric = "accuracy"
    # Set reasonable optimization parameters
    tuner.config.n_trials = 50  # More trials for better optimization
    tuner.config.cv_folds = 5   # 5-fold cross-validation
    tuner.config.verbose = 2    # Show detailed progress

    # Run tuning with descriptive task description
    results = tuner.tune(
        max_iterations=3,  # Allow multiple refinement iterations
        user_description=(
            "Multi-class classification task to predict penguin species (Adelie, Chinstrap, Gentoo) "
            "using LightGBM. The dataset contains penguin measurements including bill dimensions, "
            "flipper length, body mass, island, and sex. The pipeline includes comprehensive "
            "preprocessing with polynomial feature engineering, feature selection for classification, "
            "and LightGBM modeling optimized for multi-class prediction."
        )
    )
    
    return results


def example_with_random_forest_titanic_survival():
    """Example using Random Forest for binary classification with the Titanic dataset.
    
    This example demonstrates how to use the MLflowAutoTuner with Random Forest
    for a binary classification task to predict passenger survival on the Titanic.
    The function includes comprehensive preprocessing, feature engineering, and
    multiple pipeline steps for optimal model performance.
    
    Returns:
        dict: Results from the tuning process including best configuration,
              scores, and other metadata.
              
    Example:
        >>> results = example_with_random_forest_titanic_survival()
        >>> print(f"Best accuracy: {results['best_score']:.4f}")
    """
    print("\n" + "=" * 60)
    print("Example 8: Random Forest Binary Classification - Titanic Survival Prediction") 
    print("=" * 60)

    # Load the Titanic dataset
    import seaborn as sns
    titanic_df = sns.load_dataset('titanic')
    print("Successfully loaded Titanic dataset from seaborn")
        
    # Use 'survived' as target for binary classification
    target_col = 'survived'
    
    # Define feature categories for comprehensive preprocessing
    # Numerical features that need scaling and imputation
    numerical_features = ['age', 'fare', 'sibsp', 'parch']
    
    # Categorical features that need encoding
    categorical_features = ['sex', 'embarked', 'class', 'who', 'deck', 'embark_town']
    
    # Boolean features that can be used as-is or converted
    boolean_features = ['adult_male', 'alone']
    
    print(f"Numerical features: {numerical_features}")
    print(f"Categorical features: {categorical_features}")
    print(f"Boolean features: {boolean_features}")
    
    # Create comprehensive preprocessing pipeline
    # Handle numerical features: imputation, scaling, and polynomial features
    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing ages and fares
        ('scaler', RobustScaler()),  # Robust to outliers in fare
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True))
    ])
    
    # Handle categorical features: imputation and encoding
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing embarked/deck
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Handle boolean features: convert to int and scale
    boolean_transformer = Pipeline([
        ('converter', 'passthrough'),  # Keep as-is, they're already 0/1
    ])
    
    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bool', boolean_transformer, boolean_features)
        ],
        remainder='drop',  # Drop any remaining columns (like 'alive')
        verbose_feature_names_out=True
    )
    
    # Create Random Forest classifier with reasonable defaults for binary classification
    rf_classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced',  # Handle class imbalance
        bootstrap=True,  # Enable bootstrapping for better generalization
        oob_score=True,  # Out-of-bag scoring for additional validation
    )
    
    # Create comprehensive pipeline with multiple feature engineering steps
    pipeline = Pipeline([
        # Step 1: Comprehensive preprocessing
        ('preprocessor', preprocessor),
        
        # Step 2: Feature scaling after preprocessing (some algorithms benefit from this)
        ('feature_scaler', MinMaxScaler()),
        
        # Step 3: Feature selection using statistical tests
        ('feature_selection_univariate', SelectKBest(score_func=f_classif, k=20)),
        
        # Step 4: Dimensionality reduction for noise reduction
        ('pca', PCA(n_components=0.95, random_state=42)),  # Keep 95% of variance
        
        # Step 5: Recursive feature elimination with Random Forest
        ('feature_selection_rfe', RFE(
            estimator=RandomForestClassifier(n_estimators=10, random_state=42),
            n_features_to_select=15,
            step=1
        )),
        
        # Step 6: Final Random Forest classifier
        ('classifier', rf_classifier)
    ])

    # Initialize tuner with custom configuration
    tuner = MLflowAutoTuner(
        dataset=titanic_df,
        config_path="src/tune/config.yml", 
        metric="roc_auc",  # Use ROC-AUC for binary classification
        experiment_name="autotuner_example_random_forest_titanic_survival",
        custom_estimator=pipeline,
    )

    # Configure for binary classification task
    tuner.config.task_type = "classification"
    tuner.config.metric = "roc_auc"
    tuner.deps.target_column = target_col
    tuner.config.target_column = target_col
    tuner.config.estimator_type = "random_forest"
    tuner.deps.estimator_type = "random_forest"
    tuner.deps.task_type = "classification"
    tuner.config.direction = "maximize"  # Maximize ROC-AUC
    tuner.deps.direction = "maximize"
    tuner.deps.metric = "roc_auc"
    
    # Set optimization parameters for thorough search
    tuner.config.n_trials = 75  # More trials for comprehensive optimization
    tuner.config.cv_folds = 5   # 5-fold cross-validation for robust evaluation
    tuner.config.verbose = 2    # Show detailed progress

    # Run tuning with comprehensive task description
    results = tuner.tune(
        max_iterations=3,  # Allow multiple refinement iterations
        user_description=(
            "Binary classification task to predict passenger survival on the RMS Titanic "
            "using Random Forest. The dataset contains passenger information including "
            "ticket class, sex, age, number of siblings/spouses, number of parents/children, "
            "fare paid, port of embarkation, and other demographic features. "
            "The comprehensive pipeline includes missing value imputation, robust scaling, "
            "polynomial feature engineering, one-hot encoding for categorical variables, "
            "univariate feature selection, PCA for dimensionality reduction, "
            "recursive feature elimination, and Random Forest classification optimized "
            "for binary prediction with class balancing to handle survival rate imbalance."
        )
    )
    
    return results


def main():
    """Run all examples."""
    print("MLflow AutoTuner Examples")
    print("=" * 60)

    try:
        # # Example 1: Full API
        # example_full_api()

        # # Example 2: Convenience function
        # example_convenience_function()

        # Example 3: Custom configuration
        # example_with_custom_config()

        # # Example 4: Pipeline integration
        # example_integration_with_existing_pipeline()

        # Example 5: LightGBM Classification
        # example_with_lightgbm_classification()
        
        # Example 6: Random Forest Titanic Survival
        example_with_random_forest_titanic_survival()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
