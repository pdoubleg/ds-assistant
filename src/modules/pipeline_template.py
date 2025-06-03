import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from typing import Optional, Dict, Any

from src.modules.fe import CAAFETransformer
from src.modules.xgb_tune import XGBoostTuner  # Adjust import if needed

from sklearn.metrics import roc_auc_score, accuracy_score


def build_full_pipeline(
    df: pd.DataFrame,
    target_col: str,
    dataset_description: str,
    caafe_kwargs: Optional[Dict[str, Any]] = None,
    tuner_kwargs: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    tuner_max_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Build and run a full ML pipeline with CAAFETransformer, preprocessing, XGBoost tuning (using CV), and holdout evaluation.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target.
        target_col (str): Name of the target column.
        dataset_description (str): Description for CAAFETransformer.
        caafe_kwargs (dict, optional): Extra kwargs for CAAFETransformer.
        tuner_kwargs (dict, optional): Extra kwargs for XGBoostTuner.
        test_size (float, optional): Fraction for final holdout test set.
        random_state (int, optional): Random seed.
        tuner_max_iterations (int, optional): Max iterations for XGBoostTuner.tune().

    Returns:
        Dict[str, Any]: Dictionary containing pipeline objects and holdout metrics.

    Example:
        >>> results = build_full_pipeline(
        ...     df, "target", "Binary classification dataset",
        ...     caafe_kwargs={"iterations": 5},
        ...     tuner_kwargs={"top_n_configs": 5},
        ...     tuner_max_iterations=10
        ... )
        >>> print(results["auc"])
        >>> print(results["acc"])
    """
    caafe_kwargs = caafe_kwargs or {}
    tuner_kwargs = tuner_kwargs or {}

    # 1. Split into train/holdout only
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )

    # 2. Separate features/target
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]

    # 3. Fit CAAFETransformer on train
    caafe = CAAFETransformer(
        target_name=target_col,
        dataset_description=dataset_description,
        **caafe_kwargs
    )
    caafe.fit(X_train, y_train)

    # 4. Transform train and test with CAAFE
    X_train_fe = caafe.transform(X_train)
    X_test_fe = caafe.transform(X_test)

    # 5. Identify column types for preprocessing
    num_cols = X_train_fe.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()

    # 6. Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ]
    )

    # 7. Fit preprocessor on train, transform train and test
    preprocessor.fit(X_train_fe)
    X_train_proc = preprocessor.transform(X_train_fe)
    X_test_proc = preprocessor.transform(X_test_fe)

    # 8. Recombine preprocessed X_train_proc and y_train into a DataFrame for XGBoostTuner
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=
        preprocessor.get_feature_names_out()
    )
    train_proc_df = X_train_proc_df.copy()
    train_proc_df[target_col] = y_train.reset_index(drop=True)

    # 9. Initialize and run XGBoostTuner (uses CV internally)
    tuner = XGBoostTuner(
        dataset=train_proc_df,
        target=target_col,
        **tuner_kwargs
    )
    tuner.tune(max_iterations=tuner_max_iterations)

    # 10. Get best config/model and evaluate on holdout
    best_config = tuner.get_best_config()
    import xgboost as xgb
    final_model = xgb.XGBClassifier(**best_config)
    final_model.fit(X_train_proc, y_train)
    test_preds = final_model.predict_proba(X_test_proc)[:, 1]
    auc = roc_auc_score(y_test, test_preds)
    acc = accuracy_score(y_test, (test_preds > 0.5).astype(int))
    train_auc = roc_auc_score(y_train, final_model.predict_proba(X_train_proc)[:, 1])
    train_acc = accuracy_score(y_train, (final_model.predict(X_train_proc) > 0.5).astype(int))
    
    print(f"Train ROC AUC: {train_auc:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Holdout ROC AUC: {auc:.4f}")
    print(f"Holdout Accuracy: {acc:.4f}")

    # Optionally return objects for further use
    return {
        "caafe": caafe,
        "preprocessor": preprocessor,
        "tuner": tuner,
        "final_model": final_model,
        "X_test": X_test_proc,
        "y_test": y_test,
        "auc": auc,
        "acc": acc,
        "train_auc": train_auc,
        "train_acc": train_acc,
        "best_config": best_config,
    } 