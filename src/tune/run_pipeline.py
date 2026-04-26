import os
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from logging import getLogger
import optuna

from databricks.mlflow.utils import (
    add_group_to_experiment,
    add_group_to_registered_model,
    check_dependencies,
    log_standard_tags,
)  # type: ignore
from databricks.mlflow.utils import (
    split_data_on_indexes,
    get_split_indexes,
    get_feature_pipeline,
    get_model_pipeline,
)  # type: ignore
from databricks.mlflow.utils import evaluation_metrics  # type: ignore


logger = getLogger(__name__)


def load_data():
    """Load the data from the CSV file."""
    df = pd.read_csv("data/train.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def train_and_log_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_hyperparameters: dict[str] = None,
    feature_hyperparameters: dict[str] = None,
    split_hyperparameters: dict[str] = None,
) -> dict:
    """Run an ML Flow experiment and log to Databricks using the args sent in.

    Args:
        X (pd.DataFrame): Preprocessed data ready for training.
        y (pd.Series): Target variable for training.
        model_hyperparameters (dict[str]): Hyperparameters for the model. Default None.
        feature_hyperparameters (dict[str]): Hyperparameters for the feature engineering pipeline. Default None
        split_hyperparameters (dict[str]): Hyperparameters for how to split the data. Default: None

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

    with mlflow.start_run(nested=True):
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
            metrics = evaluation_metrics(y_test, y_pred)  # Get all metrics
            mlflow.log_metrics(metrics)  # Log all metrics
            return metrics  # Return all metrics


def objective(trial):
    """Objective function for Optuna to optimize."""
    # Suggest hyperparameters for the model
    model_hyperparameters = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),  # Number of trees
        "max_depth": trial.suggest_int("max_depth", 1, 30),  # Maximum depth of the tree
        "min_samples_split": trial.suggest_int(
            "min_samples_split", 2, 10
        ),  # Minimum samples to split
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 1, 4
        ),  # Minimum samples at a leaf node
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        ),  # Number of features to consider
    }

    feature_hyperparameters = {
        "feature_param1": trial.suggest_categorical("feature_param1", [True, False])
    }
    # Split hyperparameters if needed
    split_hyperparameters = {"cross_validate_flag": False}

    # Load data
    x, y = load_data()

    # Train the model and get the evaluation metrics
    metrics = train_and_log_model(
        x,
        y,
        model_hyperparameters=model_hyperparameters,
        feature_hyperparameters=feature_hyperparameters,
        split_hyperparameters=split_hyperparameters,
    )

    # Return the F1 score to minimize (or you could change this to another metric if needed)

    return metrics["f1"]  # Return F1 score for Optuna optimization


def run_pipeline() -> None:
    """Train and log the model to MLFlow with Optuna hyperparameter tuning."""
    # Create an Optuna study
    mlflow.set_experiment(experiment_name=os.getenv("DATABRICKS_EXPERIMENT_NAME"))

    with mlflow.start_run():
        study = optuna.create_study(
            direction="maximize"
        )  # Adjust direction if necessary
        study.optimize(objective, n_trials=3)  # You can adjust n_trials as needed
        trial = study.best_trial
        mlflow.log_params(trial.params)
        log_standard_tags()


if __name__ == "__main__":
    logger.info("Running training...")
    run_pipeline()
