import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Union, Callable
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, chi2, f_classif
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, accuracy_score, roc_auc_score
import optuna
import xgboost as xgb
import lightgbm as lgb
from dataclasses import dataclass, field
from src.utils import metric_ppv

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class FeatureSelectionResult:
    """Stores results from a feature selection method."""
    method_name: str
    selected_features: List[str]
    scores: Optional[Dict[str, float]] = None
    ranking: Optional[Dict[str, int]] = None
    parameters: Optional[Dict[str, any]] = None
    metrics: Optional[Dict[str, float]] = None


def create_estimator(
    estimator_type: str = "xgboost",
    custom_estimator: Optional[BaseEstimator] = None,
    random_state: int = 42,
) -> BaseEstimator:
    """
    Create a model instance based on the estimator type or custom estimator.
    
    Args:
        estimator_type: Type of estimator ('xgboost', 'lightgbm', or 'custom')
        custom_estimator: Custom estimator instance
        random_state: Random seed for reproducibility
        
    Returns:
        BaseEstimator: Model instance
    """
    if custom_estimator is not None:
        return custom_estimator
    if estimator_type == "xgboost":
        return xgb.XGBClassifier(
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            use_label_encoder=False,
            verbosity=0,
        )
    elif estimator_type == "lightgbm":
        return lgb.LGBMClassifier(
            random_state=random_state,
            objective="binary",
            metric="binary_logloss",
            n_jobs=-1,
            verbosity=-1,
            force_row_wise=True,
        )
    elif estimator_type == "custom":
        if custom_estimator is None:
            raise ValueError("custom_estimator must be provided for 'custom'")
        return custom_estimator
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}")


def get_scoring_function(scoring: Union[str, Callable, None], top_p: float = 0.05) -> Callable:
    """
    Get the appropriate scoring function based on the scoring parameter.
    
    Args:
        scoring: String identifier ('ppv', 'auc', 'accuracy') or callable
        top_p: Parameter for PPV calculation
        
    Returns:
        Callable scoring function
    """
    if scoring is None:
        return make_scorer(precision_score, pos_label=1)
    elif isinstance(scoring, str):
        if scoring == "ppv":
            return make_scorer(metric_ppv, top_p=top_p)
        elif scoring == "auc":
            return make_scorer(roc_auc_score)
        elif scoring == "accuracy":
            return make_scorer(accuracy_score)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")
    else:
        return scoring


class FeatureSelectionOptuna:
    """
    This class implements feature selection using Optuna optimization framework.

    Parameters:
        estimator_type (str): Type of estimator to use ('xgboost', 'lightgbm', or 'custom')
        custom_estimator (BaseEstimator, optional): Custom estimator to use if estimator_type is 'custom'
        scoring (Union[str, Callable]): The scoring metric to use. Can be:
            - 'ppv': Positive Predictive Value at top_p%
            - 'auc': Area Under ROC Curve
            - 'accuracy': Classification Accuracy
            - callable: Custom scoring function
        features (list of str): A list containing the names of all possible features that can be selected for the model.
        X (DataFrame): The complete set of feature data (pandas DataFrame) from which subsets will be selected for training the model.
        y (Series): The target variable associated with the X data (pandas Series).
        splits (list of tuples): A list of tuples where each tuple contains two elements, the train indices and the validation indices.
        top_p (float): Parameter for PPV calculation when scoring='ppv'
        random_state (int): Random seed for reproducibility
    """

    def __init__(
        self,
        estimator_type: str = "xgboost",
        custom_estimator: Optional[BaseEstimator] = None,
        scoring: Union[str, Callable] = "ppv",
        features: List[str] = None,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        splits: List[tuple] = None,
        top_p: float = 0.05,
        random_state: int = 42,
    ):
        self.estimator_type = estimator_type
        self.custom_estimator = custom_estimator
        self.model = create_estimator(estimator_type, custom_estimator, random_state)
        self.scoring_fn = get_scoring_function(scoring, top_p)
        self.features = features
        self.X = X
        self.y = y
        self.splits = splits
        self.top_p = top_p
        self.random_state = random_state

    def __call__(self, trial: optuna.trial.Trial):
        # Select True / False for each feature
        selected_features = [
            trial.suggest_categorical(name, [True, False]) for name in self.features
        ]

        # List with names of selected features
        selected_feature_names = [
            name for name, selected in zip(self.features, selected_features) if selected
        ]

        if not selected_feature_names:  # Ensure at least one feature is selected
            return float('-inf')

        score = 0

        for split in self.splits:
            train_idx = split[0]
            valid_idx = split[1]

            X_train = self.X.iloc[train_idx].copy()
            y_train = self.y.iloc[train_idx].copy()
            X_valid = self.X.iloc[valid_idx].copy()
            y_valid = self.y.iloc[valid_idx].copy()

            X_train_selected = X_train[selected_feature_names].copy()
            X_valid_selected = X_valid[selected_feature_names].copy()

            # Create fresh model instance for each fold
            model = create_estimator(self.estimator_type, self.custom_estimator, self.random_state)
            
            # Train model and get predictions
            model.fit(X_train_selected, y_train)
            
            # Use the scoring function
            if isinstance(self.scoring_fn, str):
                # Handle string scoring functions
                if self.scoring_fn == "ppv":
                    probs = model.predict_proba(X_valid_selected)
                    score += metric_ppv(y_valid, probs[:, 1], top_p=self.top_p)
                elif self.scoring_fn == "auc":
                    probs = model.predict_proba(X_valid_selected)
                    score += roc_auc_score(y_valid, probs[:, 1])
                elif self.scoring_fn == "accuracy":
                    preds = model.predict(X_valid_selected)
                    score += accuracy_score(y_valid, preds)
            else:
                # Handle callable scoring functions
                score += self.scoring_fn(model, X_valid_selected, y_valid)

        # Take the average score across all splits
        score /= len(self.splits)

        return score


class AutoFeatureSelection:
    """
    A comprehensive feature selection class that implements multiple selection methods
    and provides easy access to results and feature intersections.

    Parameters:
        estimator_type (str): Type of estimator to use ('xgboost', 'lightgbm', or 'custom')
        custom_estimator (BaseEstimator, optional): Custom estimator to use if estimator_type is 'custom'
        scoring (Union[str, Callable]): Scoring metric to use. Can be:
            - 'ppv': Positive Predictive Value at top_p%
            - 'auc': Area Under ROC Curve
            - 'accuracy': Classification Accuracy
            - callable: Custom scoring function
        n_splits (int): Number of cross-validation splits
        random_state (int): Random seed for reproducibility
        top_p (float): Parameter for PPV metric calculation

    Methods:
        run_optuna_selection: Performs feature selection using Optuna optimization
        run_kbest_selection: Performs selection using SelectKBest
        run_sequential_selection: Performs sequential feature selection
        run_model_based_selection: Performs selection using model's feature importance
        get_feature_intersection: Gets features selected by multiple methods
        get_top_n_consistent: Gets top N features that rank highly across methods
    """

    def __init__(
        self,
        estimator_type: str = "xgboost",
        custom_estimator: Optional[BaseEstimator] = None,
        scoring: Union[str, Callable] = "ppv",
        n_splits: int = 5,
        random_state: int = 42,
        top_p: float = 0.05,
    ):
        if estimator_type == "custom" and custom_estimator is None:
            raise ValueError("custom_estimator must be provided for 'custom'")
        if estimator_type not in ("xgboost", "lightgbm", "custom"):
            raise ValueError(f"Unsupported estimator_type: {estimator_type}")
            
        self.estimator_type = estimator_type
        self.custom_estimator = custom_estimator
        self.model = create_estimator(estimator_type, custom_estimator, random_state)
        self.scoring = scoring
        self.scoring_fn = get_scoring_function(scoring, top_p)
        self.n_splits = n_splits
        self.random_state = random_state
        self.top_p = top_p
        
        # Store results from each method
        self.results: Dict[str, FeatureSelectionResult] = {}
        
        # Track all features seen
        self.all_features: Set[str] = set()

    def run_optuna_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 500,
        timeout: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """
        Performs feature selection using Optuna optimization.
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            
        Returns:
            FeatureSelectionResult containing selected features and metrics
        """
        features = list(X.columns)
        self.all_features.update(features)
        
        # Create cross-validation splits
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        splits = list(skf.split(X, y))
        
        # Create and configure the study
        study = optuna.create_study(direction="maximize")
        
        # Configure the optimization
        fs_optuna = FeatureSelectionOptuna(
            estimator_type=self.estimator_type,
            custom_estimator=self.custom_estimator,
            scoring=self.scoring,
            features=features,
            X=X,
            y=y,
            splits=splits,
            top_p=self.top_p,
            random_state=self.random_state,
        )
        
        # Run optimization
        study.optimize(fs_optuna, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # Get selected features from best trial
        selected_features = [
            feat for feat, selected in zip(features, study.best_trial.params.values())
            if selected
        ]
        
        # Store results
        result = FeatureSelectionResult(
            method_name="optuna",
            selected_features=selected_features,
            parameters=study.best_trial.params,
            metrics={"best_score": study.best_trial.value}
        )
        self.results["optuna"] = result
        
        return result

    def run_kbest_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: Union[int, str] = "all",
        score_func: callable = f_classif,
    ) -> FeatureSelectionResult:
        """
        Performs feature selection using SelectKBest.
        
        Args:
            X: Feature DataFrame
            y: Target series
            k: Number of features to select
            score_func: Scoring function to use
            
        Returns:
            FeatureSelectionResult containing selected features and scores
        """
        features = list(X.columns)
        self.all_features.update(features)
        
        # Initialize and fit SelectKBest
        skb = SelectKBest(score_func=score_func, k=k)
        skb.fit(X, y)
        
        # Get scores and selected features
        scores = dict(zip(features, skb.scores_))
        selected_features = [
            feat for feat, selected in zip(features, skb.get_support())
            if selected
        ]
        
        # Create ranking based on scores
        ranking = dict(zip(
            features,
            [sorted(scores.values(), reverse=True).index(score) + 1 for score in scores.values()]
        ))
        
        # Store results
        result = FeatureSelectionResult(
            method_name="kbest",
            selected_features=selected_features,
            scores=scores,
            ranking=ranking
        )
        self.results["kbest"] = result
        
        return result

    def run_sequential_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_to_select: Union[int, float] = 0.5,
        direction: str = "forward",
    ) -> FeatureSelectionResult:
        """
        Performs sequential feature selection.
        
        Args:
            X: Feature DataFrame
            y: Target series
            n_features_to_select: Number of features to select
            direction: 'forward' or 'backward' selection
            
        Returns:
            FeatureSelectionResult containing selected features
        """
        features = list(X.columns)
        self.all_features.update(features)
        
        # Create cross-validation splits
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Initialize and fit SFS
        sfs = SequentialFeatureSelector(
            self.model,
            n_features_to_select=n_features_to_select,
            direction=direction,
            cv=cv,
        )
        sfs.fit(X, y)
        
        # Get selected features
        selected_features = list(X.columns[sfs.get_support()])
        
        # Store results
        result = FeatureSelectionResult(
            method_name="sequential",
            selected_features=selected_features,
            parameters={"direction": direction, "n_features": n_features_to_select}
        )
        self.results["sequential"] = result
        
        return result

    def run_model_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        importance_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> FeatureSelectionResult:
        """
        Performs feature selection using model's feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target series
            importance_threshold: Minimum importance threshold
            top_n: Number of top features to select
            
        Returns:
            FeatureSelectionResult containing selected features and importance scores
        """
        features = list(X.columns)
        self.all_features.update(features)
        
        # Fit model
        self.model.fit(X, y)
        
        # Get feature importance scores
        if hasattr(self.model, "feature_importances_"):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance_scores = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")
        
        # Create scores dictionary
        scores = dict(zip(features, importance_scores))
        
        # Select features based on criteria
        if importance_threshold is not None:
            selected_features = [
                feat for feat, score in scores.items()
                if score >= importance_threshold
            ]
        elif top_n is not None:
            selected_features = [
                feat for feat, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            ]
        else:
            selected_features = features
            
        # Create ranking based on scores
        ranking = dict(zip(
            features,
            [sorted(scores.values(), reverse=True).index(score) + 1 for score in scores.values()]
        ))
        
        # Store results
        result = FeatureSelectionResult(
            method_name="model_based",
            selected_features=selected_features,
            scores=scores,
            ranking=ranking,
            parameters={"threshold": importance_threshold, "top_n": top_n}
        )
        self.results["model_based"] = result
        
        return result

    def get_feature_intersection(self, methods: Optional[List[str]] = None) -> List[str]:
        """
        Gets features that were selected by all specified methods.
        
        Args:
            methods: List of method names to consider. If None, uses all methods.
            
        Returns:
            List of features selected by all specified methods
        """
        if not methods:
            methods = list(self.results.keys())
            
        if not all(method in self.results for method in methods):
            raise ValueError(f"Some methods not yet run. Available methods: {list(self.results.keys())}")
            
        # Get sets of selected features for each method
        method_features = [
            set(self.results[method].selected_features)
            for method in methods
        ]
        
        # Return intersection
        return list(set.intersection(*method_features))

    def get_top_n_consistent(self, n: int = 10, methods: Optional[List[str]] = None) -> List[str]:
        """
        Gets top N features that rank consistently high across methods.
        
        Args:
            n: Number of features to return
            methods: List of method names to consider. If None, uses all methods with rankings.
            
        Returns:
            List of top N consistently high-ranking features
        """
        if not methods:
            methods = [
                method for method, result in self.results.items()
                if result.ranking is not None
            ]
            
        if not methods:
            raise ValueError("No methods with rankings available")
            
        # Get all rankings
        all_rankings = []
        for method in methods:
            if self.results[method].ranking is None:
                continue
            all_rankings.append(self.results[method].ranking)
            
        # Calculate average ranking for each feature
        avg_rankings = {}
        for feature in self.all_features:
            rankings = [
                ranking.get(feature, len(self.all_features))  # Use worst rank if feature not in ranking
                for ranking in all_rankings
            ]
            avg_rankings[feature] = np.mean(rankings)
            
        # Return top N features by average ranking
        return [
            feature for feature, _ in sorted(avg_rankings.items(), key=lambda x: x[1])[:n]
        ]

    def get_summary(self) -> Dict[str, any]:
        """
        Returns a summary of all feature selection results.
        
        Returns:
            Dictionary containing summary statistics and results
        """
        return {
            "total_features": len(self.all_features),
            "methods_used": list(self.results.keys()),
            "features_per_method": {
                method: len(result.selected_features)
                for method, result in self.results.items()
            },
            "intersection_features": self.get_feature_intersection(),
            "results": self.results
        }
