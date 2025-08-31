"""
Enhanced MLflow pipeline with AutoTuner integration.

This module demonstrates how to integrate the MLflowAutoTuner with existing MLflow
pipelines, replacing manual hyperparameter optimization with LLM-guided tuning.

Key Features:
- Drop-in replacement for manual Optuna optimization
- Agent-based logging instead of trial-by-trial logging
- Natural integration with existing pipeline structure
- Configuration-driven tuning workflow

Author: Generated for MLflow AutoTuner integration
"""

import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from logging import getLogger

try:
    import mlflow
except ImportError:
    mlflow = None

from databricks.mlflow.utils import add_group_to_experiment, add_group_to_registered_model, check_dependencies, log_standard_tags # type: ignore
from databricks.mlflow.utils import split_data_on_indexes, get_split_indexes, get_feature_pipeline, get_model_pipeline # type: ignore
from databricks.mlflow.utils import evaluation_metrics # type: ignore

# Import our MLflow AutoTuner
from .tune_mlflow import MLflowAutoTuner, run_mlflow_tuning

logger = getLogger(__name__)


def load_data():
    """Load the data from the CSV file."""
    df = pd.read_csv("data/train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train_and_log_model_with_best_params(
    X: pd.DataFrame,
    y: pd.Series,
    model_hyperparameters: dict = None,
    feature_hyperparameters: dict = None,
    split_hyperparameters: dict = None,
) -> dict:
    """
    Train and log model using the best hyperparameters from AutoTuner.
    
    Args:
        X (pd.DataFrame): Preprocessed data ready for training.
        y (pd.Series): Target variable for training.
        model_hyperparameters (dict): Best hyperparameters from AutoTuner.
        feature_hyperparameters (dict): Feature engineering hyperparameters.
        split_hyperparameters (dict): Data splitting hyperparameters.

    Returns:
        dict: A dictionary containing evaluation metrics of the model.
    """
    mlflow.set_experiment(experiment_name=os.getenv("DATABRICKS_EXPERIMENT_NAME"))
    experiment = mlflow.get_experiment_by_name(os.getenv("DATABRICKS_EXPERIMENT_NAME"))
    add_group_to_experiment(experiment_id=str(experiment.experiment_id))
    mlflow.sklearn.autolog(log_datasets=False, log_models=False, exclusive=False)
    reqs = check_dependencies()

    log_standard_tags()

    splits = get_split_indexes(X, y, **split_hyperparameters)

    with mlflow.start_run(nested=True, run_name="final_model_training"):
        for idx, split in enumerate(splits):
            logger.info("Getting train/test splits")
            train_indexes, test_indexes = split
            mlflow.log_param(f"split_{idx}_train_indexes", train_indexes)
            mlflow.log_param(f"split_{idx}_test_indexes", test_indexes)
            X_train, X_test, y_train, y_test = split_data_on_indexes(
                X, y, [train_indexes, test_indexes]
            )

            features = get_feature_pipeline(feature_hyperparameters)
            if not isinstance(features, BaseEstimator):
                features = FunctionTransformer(features)
            
            model = get_model_pipeline(**model_hyperparameters)

            pipeline = Pipeline(
                steps=[("features", features), ("model", model)], verbose=True
            )

            registered_model_name = (
                os.getenv("DATABRICKS_REGISTERED_MODEL_NAME")
                if os.getenv("DATABRICKS_REGISTERED_MODEL_NAME") != "None"
                else None
            )

            pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=(registered_model_name),
                pip_requirements=reqs,
            )
            if registered_model_name is not None:
                add_group_to_registered_model(
                    registered_model_name=registered_model_name
                )

            y_pred = pipeline.predict(X_test)

            # Calculate and log evaluation metrics
            metrics = evaluation_metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)
            return metrics


def run_autotuner_optimization(
    X: pd.DataFrame, 
    y: pd.Series,
    config_path: str = "src/tune/config.yml",
    metric: str = "f1",
) -> dict:
    """
    Run AutoTuner optimization to find best hyperparameters.
    
    Args:
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable.
        config_path (str): Path to AutoTuner configuration file.
        metric (str): Optimization metric.
        
    Returns:
        dict: Best hyperparameters found by AutoTuner.
    """
    # Create dataset for AutoTuner
    dataset = X.copy()
    dataset['target'] = y
    
    # Run MLflow AutoTuner
    logger.info("Starting AutoTuner hyperparameter optimization...")
    
    tuner = MLflowAutoTuner(
        config_path=config_path,
        dataset=dataset,
        metric=metric,
        experiment_name=os.getenv("DATABRICKS_EXPERIMENT_NAME")
    )
    
    # Run tuning - this will log everything to MLflow
    tuning_results = tuner.tune()
    
    # Extract best hyperparameters
    best_config = tuner.get_best_config()
    
    if best_config is None:
        logger.warning("No valid configuration found, using defaults")
        return {}
    
    logger.info(f"AutoTuner found best configuration with score: {tuning_results['best_score']:.4f}")
    logger.info(f"Best hyperparameters: {best_config}")
    
    return best_config


def run_pipeline_with_autotuner() -> None:
    """
    Enhanced pipeline that uses AutoTuner for hyperparameter optimization.
    
    This replaces the manual Optuna optimization with LLM-guided AutoTuner,
    providing more intelligent hyperparameter search and better MLflow integration.
    """
    mlflow.set_experiment(experiment_name=os.getenv("DATABRICKS_EXPERIMENT_NAME"))
    
    with mlflow.start_run(run_name="autotuner_pipeline"):
        log_standard_tags()
        
        # Load data
        logger.info("Loading data...")
        X, y = load_data()
        
        # Run AutoTuner optimization
        logger.info("Running AutoTuner hyperparameter optimization...")
        best_hyperparameters = run_autotuner_optimization(
            X, y,
            config_path="src/tune/config.yml",
            metric="f1"  # Optimization metric
        )
        
        # Log the best hyperparameters from AutoTuner
        mlflow.log_params(best_hyperparameters)
        
        # Feature and split hyperparameters (from your existing pipeline)
        feature_hyperparameters = {
            "feature_param1": True  # Example feature parameter
        }
        split_hyperparameters = {"cross_validate_flag": False}
        
        # Train final model with best hyperparameters
        logger.info("Training final model with optimized hyperparameters...")
        final_metrics = train_and_log_model_with_best_params(
            X,
            y,
            model_hyperparameters=best_hyperparameters,
            feature_hyperparameters=feature_hyperparameters,
            split_hyperparameters=split_hyperparameters,
        )
        
        # Log final metrics at the top level
        mlflow.log_metrics(final_metrics)
        
        logger.info(f"Pipeline completed successfully. Final F1 score: {final_metrics.get('f1', 'N/A')}")


def run_pipeline_simple_integration() -> None:
    """
    Simplified integration example using the convenience function.
    
    This shows the most straightforward way to integrate AutoTuner
    into an existing pipeline with minimal code changes.
    """
    mlflow.set_experiment(experiment_name=os.getenv("DATABRICKS_EXPERIMENT_NAME"))
    
    with mlflow.start_run(run_name="simple_autotuner_integration"):
        log_standard_tags()
        
        # Load data
        X, y = load_data()
        
        # Create dataset for AutoTuner
        dataset = X.copy()
        dataset['target'] = y
        
        # Run tuning with convenience function
        tuning_results = run_mlflow_tuning(
            dataset=dataset,
            config_path="src/tune/config.yml",
            metric="f1",
            experiment_name=os.getenv("DATABRICKS_EXPERIMENT_NAME")
        )
        
        # Extract best parameters and train final model
        best_params = tuning_results.get('best_config', {})
        
        if best_params:
            final_metrics = train_and_log_model_with_best_params(
                X, y,
                model_hyperparameters=best_params,
                feature_hyperparameters={"feature_param1": True},
                split_hyperparameters={"cross_validate_flag": False},
            )
            mlflow.log_metrics(final_metrics)
            logger.info(f"Final model F1 score: {final_metrics.get('f1', 'N/A')}")



if __name__ == "__main__":
    logger.info("Running enhanced pipeline with AutoTuner...")
    
    # Choose which version to run:
    # run_pipeline_with_autotuner()  # Full integration
    run_pipeline_simple_integration()  # Simple integration
    # run_pipeline()  # Original (deprecated)
