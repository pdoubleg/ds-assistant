# GeneralizedTuner: Flexible AutoML Hyperparameter Optimization

The `AutoTuneLLM` class provides a flexible, LLM-guided hyperparameter optimization framework that supports XGBoost, LightGBM, and custom sklearn estimators. It offers two distinct workflows to accommodate different use cases and data privacy requirements.

## Features

### ✨ Key Capabilities

- **Multi-Estimator Support**: XGBoost, LightGBM, and custom sklearn estimators
- **Flexible Scoring**: Custom scoring functions or sklearn metric strings
- **Dual Workflows**: Dataset-aware and user-description-based optimization
- **Intelligent Search Space Refinement**: LLM-guided iterative optimization
- **Early Stopping**: Automatic termination when no improvement is detected
- **Best Configuration Tracking**: Maintains top-N configurations across iterations

### 🔧 Supported Estimators

- **XGBoost**: `estimator_type="xgboost"`
- **LightGBM**: `estimator_type="lightgbm"`
- **Custom**: `estimator_type="custom"` with any sklearn-compatible estimator

## Installation & Setup

Ensure you have the required dependencies installed:

```bash
pip install xgboost lightgbm scikit-learn pandas numpy pydantic-ai
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Basic XGBoost Tuning with Dataset Analysis

```python
from src.modules.tune import AutoTuneLLM
from sklearn.metrics import make_scorer, precision_score
import pandas as pd

# Initialize tuner
tuner = AutoTuneLLM(
    estimator_type="xgboost",
    dataset=your_dataframe,
    target="target_column",
    scoring=make_scorer(precision_score, pos_label=1)
)

# Run tuning with dataset analysis
tuner.tune_with_dataset_analysis(max_iterations=10)

# Get results
best_config = tuner.get_best_config()
best_estimator = tuner.create_best_estimator()
```

## Workflows

### 1. Dataset-Aware Workflow

This workflow analyzes your dataset characteristics and incorporates them into the LLM prompts for more informed hyperparameter suggestions.

```python
tuner = AutoTuneLLM(
    estimator_type="lightgbm",
    dataset=df,
    target="target",
    scoring="f1",
    task_description="Multi-class classification with imbalanced data"
)

# LLM receives dataset characteristics
tuner.tune_with_dataset_analysis(max_iterations=15)
```

**Use Cases:**

- When dataset privacy is not a concern
- You want the LLM to consider actual data characteristics
- Optimal hyperparameter suggestions based on real data patterns

### 2. User-Description Workflow

This workflow relies solely on your description of the data and task, without passing any actual dataset information to the LLM.

```python
user_description = """
Binary classification problem with 50,000 samples and 100 features.
The dataset has:
- Moderate class imbalance (70/30 split)
- Mix of numerical and categorical features
- Some missing values (handled)
- Goal: Maximize recall while maintaining reasonable precision
- Need to avoid overfitting due to noise in features
"""

tuner = AutoTuneLLM(
    estimator_type="xgboost",
    dataset=df,  # Used for actual tuning, not sent to LLM
    target="target",
    scoring=make_scorer(recall_score)
)

# LLM only sees your description
tuner.tune_with_user_description(user_description, max_iterations=10)
```

**Use Cases:**

- Sensitive or proprietary datasets
- Compliance requirements (GDPR, HIPAA, etc.)
- Large datasets where sending samples is impractical
- You want full control over what information the LLM receives

## Advanced Usage

### Custom Estimators

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Random Forest
custom_rf = RandomForestClassifier(random_state=42)
tuner = GeneralizedTuner(
    estimator_type="custom",
    custom_estimator=custom_rf,
    dataset=df,
    target="target",
    scoring="accuracy"
)

# SVM
custom_svm = SVC(probability=True, random_state=42)
tuner = GeneralizedTuner(
    estimator_type="custom", 
    custom_estimator=custom_svm,
    dataset=df,
    target="target",
    scoring=make_scorer(roc_auc_score, needs_proba=True)
)
```

### Custom Scoring Functions

```python
from sklearn.metrics import make_scorer

def custom_business_metric(y_true, y_pred):
    """Custom scoring function for business objectives."""
    # Your custom logic here
    return some_business_score

# Use custom scorer
tuner = AutoTuneLLM(
    estimator_type="xgboost",
    dataset=df,
    target="target", 
    scoring=make_scorer(custom_business_metric, greater_is_better=True)
)
```

### Configuration Options

```python
tuner = AutoTuneLLM(
    estimator_type="lightgbm",
    dataset=df,
    target="target",
    scoring="f1_weighted",
    task_description="Multi-class classification with class imbalance",
    top_n_configs=10,                    # Keep top 10 configs
    max_consecutive_no_improvement=5     # Stop after 5 iterations without improvement
)
```

## API Reference

### GeneralizedTuner

#### Constructor Parameters

| Parameter                          | Type                 | Default             | Description                                         |
| ---------------------------------- | -------------------- | ------------------- | --------------------------------------------------- |
| `estimator_type`                 | str                  | "xgboost"           | Type of estimator ("xgboost", "lightgbm", "custom") |
| `custom_estimator`               | BaseEstimator        | None                | Required when estimator_type="custom"               |
| `scoring`                        | Union[str, Callable] | None                | Scoring function or sklearn metric string           |
| `dataset`                        | pd.DataFrame         | None                | Dataset for tuning                                  |
| `target`                         | str                  | None                | Target column name                                  |
| `task_description`               | str                  | Default description | Task description for LLM context                    |
| `top_n_configs`                  | int                  | 5                   | Number of top configurations to retain              |
| `max_consecutive_no_improvement` | int                  | 3                   | Early stopping threshold                            |

#### Main Methods

**`tune_with_dataset_analysis(max_iterations: int = 10)`**

- Runs dataset-aware tuning workflow
- Analyzes dataset characteristics for LLM prompts
- Requires dataset and target to be provided

**`tune_with_user_description(user_description: str, max_iterations: int = 10)`**

- Runs user-description-based tuning workflow
- Uses only user-provided task description
- Dataset not passed to LLM (but still needed for actual tuning)

**`get_best_config() -> dict`**

- Returns the best hyperparameter configuration found

**`get_best_configs(n: int = None) -> List[dict]`**

- Returns top N configurations with scores and iteration info

**`get_tuning_summary() -> dict`**

- Returns comprehensive summary of tuning process

**`create_best_estimator() -> BaseEstimator`**

- Creates an estimator configured with the best hyperparameters

## Examples

See the complete examples in `examples/generalized_tuner_example.py`:

```python
# Run examples
python examples/generalized_tuner_example.py
```

### Example Output

```
Best Configuration: {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1, ...}
Best Score: 0.8542
Total Iterations: 7
Score progression: [0.8234, 0.8367, 0.8542, 0.8539, 0.8521, 0.8542, 0.8535]
```

## Comparison with Original XGBoostTuner

| Feature           | XGBoostTuner          | GeneralizedTuner                    |
| ----------------- | --------------------- | ----------------------------------- |
| Estimator Support | XGBoost only          | XGBoost, LightGBM, Custom           |
| Scoring Function  | Fixed precision       | Configurable                        |
| Workflows         | Dataset analysis only | Dataset analysis + User description |
| Privacy Options   | No                    | Yes (user-description workflow)     |
| Extensibility     | Limited               | High                                |

## Best Practices

### 1. Choosing Workflows

**Use Dataset-Aware Workflow When:**

- Dataset privacy is not a concern
- You want optimal hyperparameter suggestions
- Dataset size is manageable for analysis

**Use User-Description Workflow When:**

- Working with sensitive data
- Compliance requirements restrict data sharing
- Large datasets where sampling is impractical
- You want precise control over LLM inputs

### 2. Iteration Planning

```python
# Start with fewer iterations for exploration
tuner.tune_with_dataset_analysis(max_iterations=5)

# If promising, continue with more iterations
if tuner.get_tuning_summary()['best_score'] > threshold:
    tuner.tune_with_dataset_analysis(max_iterations=10)
```

### 3. Custom Estimator Guidelines

- Ensure estimator is sklearn-compatible
- Provide informative task descriptions for better hyperparameter suggestions
- Test with small iterations first to validate compatibility

### 4. Scoring Function Selection

```python
# For imbalanced binary classification
scoring = make_scorer(f1_score)

# For multi-class with class weights
scoring = "f1_weighted"

# For ranking/probability tasks  
scoring = make_scorer(roc_auc_score, needs_proba=True)

# Custom business metrics
scoring = make_scorer(your_custom_function, greater_is_better=True)
```

## Troubleshooting

### Common Issues

1. **ImportError: Missing dependencies**

   ```bash
   pip install xgboost lightgbm scikit-learn
   ```
2. **API Key Error**

   ```bash
   export OPENAI_API_KEY="your-key"
   ```
3. **Custom Estimator Incompatibility**

   - Ensure estimator has `fit()` and `predict()` methods
   - Check that estimator supports `set_params()`
4. **Memory Issues with Large Datasets**

   - Use user-description workflow to avoid passing data to LLM
   - Consider data sampling for dataset-aware workflow

### Performance Tips

- Start with `max_consecutive_no_improvement=2` for quick iteration
- Use fewer `top_n_configs` for faster processing
- Consider parallel execution for multiple estimator comparison

## Integration

### With MLflow

```python
import mlflow

with mlflow.start_run():
    tuner.tune_with_dataset_analysis(max_iterations=10)
  
    # Log results
    best_config = tuner.get_best_config()
    summary = tuner.get_tuning_summary()
  
    mlflow.log_params(best_config)
    mlflow.log_metric("best_score", summary['best_score'])
    mlflow.log_metric("total_iterations", summary['total_iterations'])
  
    # Log model
    best_estimator = tuner.create_best_estimator()
    mlflow.sklearn.log_model(best_estimator, "best_model")
```

### With Optuna

```python
# Use GeneralizedTuner results as starting point for Optuna
best_config = tuner.get_best_config()

def objective(trial):
    # Use best_config to define narrower search ranges
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 
                                         best_config['n_estimators']-50,
                                         best_config['n_estimators']+50),
        # ... other parameters based on best_config
    }
    # ... rest of objective function
```

## Contributing

When extending the GeneralizedTuner:

1. **Adding New Estimators**: Update `create_estimator()` function
2. **New Scoring Methods**: Ensure compatibility with HalvingRandomSearchCV
3. **Prompt Engineering**: Test prompt changes with different estimator types
4. **Documentation**: Update this README and add examples

## License

This module is part of the ds-assistant project. See project LICENSE for details.
