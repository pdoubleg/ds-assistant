# AutoTuner - Automated ML Hyperparameter Optimization

A generalized machine learning model tuner that supports both classification and regression tasks with any sklearn-compatible estimator. Features LLM-guided search space optimization, rich CLI interface, and comprehensive result reporting.

## 🚀 Key Features

- **Universal Compatibility**: Works with any sklearn-compatible estimator
- **Task Agnostic**: Supports both classification and regression
- **LLM-Guided Optimization**: Uses GPT models to intelligently refine search spaces
- **Rich CLI Interface**: Beautiful terminal interface with progress bars and tables
- **YAML Configuration**: Highly configurable via YAML files
- **Comprehensive Metrics**: Supports all sklearn metrics plus custom functions
- **Result Persistence**: Saves models, studies, and detailed summaries
- **Interactive Mode**: User-friendly setup wizard

## 📦 Installation

The AutoTuner is part of the ds-assistant project. Make sure you have all dependencies installed:

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Or using uv
uv sync
```

## 🔧 Configuration

The tuner is configured via YAML files. See `config.yml` for all available options:

```yaml
# Data Settings
data:
  path: "data/my_dataset.csv"  # Path to your CSV data file
  target: "target_column"      # Name of target column

# General Settings
general:
  random_state: 42
  verbose: 1
  n_jobs: -1

# Cross-validation settings
cross_validation:
  cv_folds: 5
  n_repeats: 2
  test_size: 0.2
  stratify: true

# Optuna optimization settings
optuna:
  n_trials: 100
  max_iterations: 5
  max_no_improve: 3
  top_n_configs: 5
  sampler: "TPESampler"
  direction: "maximize"

# LLM settings
llm:
  model: "gpt-4o-mini"
  use_dataset_analysis: true

# And many more options...
```

## 🎯 Usage

### Command Line Interface

#### Basic Usage

```bash
# Classification with XGBoost (CLI arguments)
python src/tune/tuner.py --data data.csv --target target_col --estimator xgboost --metric accuracy

# Regression with LightGBM (CLI arguments)
python src/tune/tuner.py --data data.csv --target target_col --estimator lightgbm --metric r2 --task regression

# Using configuration file (set data.path and data.target in YAML)
python src/tune/tuner.py --config my_config.yml --estimator xgboost --metric accuracy

# Override config data with CLI args
python src/tune/tuner.py --config my_config.yml --data different_data.csv --estimator lightgbm
```

#### Interactive Mode

```bash
python src/tune/tuner.py --interactive
```

This launches an interactive wizard that guides you through:
- Data file selection
- Target column selection
- Task type detection/confirmation
- Estimator selection
- Metric selection

#### Advanced Options

```bash
# Traditional CLI approach with all options
python src/tune/tuner.py \
  --data data.csv \
  --target target_col \
  --estimator random_forest \
  --metric f1 \
  --max-iterations 10 \
  --n-trials 200 \
  --output-dir results/ \
  --config custom_config.yml

# Config-first approach with selective overrides
python src/tune/tuner.py \
  --config my_config.yml \
  --estimator xgboost \
  --max-iterations 15 \
  --output-dir experiment_1/
```

### Python API

#### Basic Example

```python
from src.tune.tuner import AutoTuner
import pandas as pd

# Load your data
df = pd.read_csv("data.csv")

# Create tuner
tuner = AutoTuner(
    dataset=df,
    target="target_column",
    estimator_type="xgboost",
    task_type="classification",  # or "regression" or "auto"
    metric="accuracy"
)

# Run tuning
tuner.tune(max_iterations=5)

# Get results
best_config = tuner.get_best_config()
summary = tuner.get_tuning_summary()
```

#### Custom Estimator Example

```python
from sklearn.ensemble import RandomForestClassifier
from src.tune.tuner import AutoTuner

# Create custom estimator
custom_rf = RandomForestClassifier(random_state=42)

# Create tuner
tuner = AutoTuner(
    dataset=df,
    target="target_column",
    estimator=custom_rf,
    estimator_type="random_forest",  # Helps with search space generation
    metric="f1"
)

tuner.tune()
```

#### Configuration-Based Example

```python
from src.tune.tuner import AutoTuner, AutoTunerConfig

# Load configuration from YAML (including data path and target)
config = AutoTunerConfig.from_yaml("custom_config.yml")

# Override specific settings
config.max_iterations = 10
config.n_trials = 200

# Create tuner - data loaded automatically from config
tuner = AutoTuner(
    config=config,
    estimator_type="lightgbm",
    metric="roc_auc"
)

# Or override config data with different dataset
import pandas as pd
df = pd.read_csv("different_data.csv")
tuner = AutoTuner(
    config=config,
    dataset=df,              # Overrides config.data_path
    target="other_target",   # Overrides config.target_column
    estimator_type="lightgbm",
    metric="roc_auc"
)

tuner.tune()
```

## 📁 Data Configuration

The AutoTuner supports flexible data specification with three approaches:

### 1. Configuration File (Recommended)

Set data path and target in your YAML config:

```yaml
# config.yml
data:
  path: "datasets/my_data.csv"
  target: "target_column"
```

Then run without data arguments:
```bash
python src/tune/tuner.py --config config.yml --estimator xgboost
```

### 2. CLI Arguments

Specify data directly via command line:
```bash
python src/tune/tuner.py --data data.csv --target target_col --estimator xgboost
```

### 3. Python API

Load data in Python and pass to tuner:
```python
import pandas as pd
df = pd.read_csv("data.csv")

tuner = AutoTuner(
    dataset=df,
    target="target_column",
    estimator_type="xgboost"
)
```

### Priority Order

When multiple data sources are specified:
1. **Python API** (`dataset` parameter) - highest priority
2. **CLI Arguments** (`--data`, `--target`) - overrides config
3. **Configuration File** (`data.path`, `data.target`) - fallback

## 🎛️ Supported Estimators

### Built-in Support
- **XGBoost**: `xgboost` (classification & regression)
- **LightGBM**: `lightgbm` (classification & regression)  
- **Random Forest**: `random_forest` (classification & regression)
- **SVM**: `svm` (classification & regression)
- **Logistic Regression**: `logistic_regression` (classification only)
- **Linear Regression**: `linear_regression` (regression only)

### Custom Estimators
Any sklearn-compatible estimator can be used by passing it directly:

```python
from sklearn.neural_network import MLPClassifier

tuner = AutoTuner(
    dataset=df,
    target="target",
    estimator=MLPClassifier(),
    estimator_type="custom",  # or let it auto-detect
    metric="accuracy"
)
```

## 📊 Supported Metrics

### Classification Metrics
- `accuracy`
- `precision`
- `recall` 
- `f1`
- `roc_auc`
- `average_precision`
- `balanced_accuracy`
- `neg_log_loss`

### Regression Metrics
- `r2`
- `neg_mean_squared_error`
- `neg_mean_absolute_error`
- `neg_root_mean_squared_error`
- `neg_mean_absolute_percentage_error`

### Custom Metrics
You can also pass custom sklearn-compatible scoring functions:

```python
from sklearn.metrics import make_scorer, fbeta_score

# Custom F-beta score
custom_scorer = make_scorer(fbeta_score, beta=2)

tuner = AutoTuner(
    dataset=df,
    target="target",
    estimator_type="xgboost",
    metric=custom_scorer
)
```

## 🧠 LLM-Guided Optimization

The AutoTuner uses large language models to intelligently guide the hyperparameter search:

1. **Dataset Analysis**: Analyzes your dataset characteristics
2. **Initial Search Space**: Generates appropriate initial hyperparameter ranges
3. **Iterative Refinement**: Adapts search space based on trial results
4. **Domain Knowledge**: Incorporates ML best practices and estimator-specific insights

### LLM Configuration

```yaml
llm:
  model: "gpt-4o-mini"  # or "gpt-4", "gpt-3.5-turbo"
  use_dataset_analysis: true
  max_retries: 3
```

## 📁 Output and Results

### Automatic Saving

When `save_results: true` in config, the tuner automatically saves:

- `tuning_summary.pkl`: Complete tuning summary with all metrics
- `best_model.pkl`: Trained model with best hyperparameters
- `optuna_studies.pkl`: All Optuna study objects for analysis

### Results Structure

```python
summary = tuner.get_tuning_summary()
# Contains:
# - best_score: Best achieved score
# - best_config: Best hyperparameter configuration  
# - top_configs: Top N configurations
# - iterations: Number of iterations run
# - progression: Score progression across iterations
# - baseline_metrics: Performance before tuning
# - final_metrics: Performance after tuning
```

## 🎨 CLI Output Features

The CLI provides rich, colorful output including:

- **Progress Bars**: Real-time optimization progress
- **Result Tables**: Formatted trial results with ranking
- **Performance Panels**: Highlighted metrics and improvements
- **Configuration Display**: Clear parameter summaries
- **Error Handling**: Helpful error messages and suggestions

## 🔄 Examples

Run the provided examples to see the tuner in action:

```bash
# Python API examples
python examples/tuner_example.py

# CLI demonstration
python examples/cli_demo.py
```

## ⚙️ Advanced Configuration

### Optuna Settings

```yaml
optuna:
  sampler: "TPESampler"  # TPESampler, RandomSampler, CmaEsSampler
  pruner: "MedianPruner"  # MedianPruner, HyperbandPruner, null
  direction: "maximize"   # maximize or minimize
```

### Cross-Validation Settings

```yaml
cross_validation:
  cv_folds: 5      # Number of CV folds
  n_repeats: 2     # Repeated CV
  test_size: 0.2   # Hold-out test size
  stratify: true   # Stratified sampling (classification)
```

### Display Settings

```yaml
display:
  show_progress_bar: true
  update_frequency: 10
  max_table_rows: 20
  decimal_precision: 4
```

## 🚨 Error Handling

The tuner includes comprehensive error handling for:

- **Invalid Data**: Missing files, wrong formats, missing columns
- **Configuration Errors**: Invalid YAML, missing required settings
- **Model Errors**: Incompatible estimators, invalid parameters
- **LLM Errors**: API failures, invalid generated code
- **Optimization Errors**: Failed trials, convergence issues

## 🔍 Troubleshooting

### Common Issues

1. **OpenAI API Key**: Make sure `OPENAI_API_KEY` environment variable is set
2. **Data Not Found**: Check file paths in config or CLI arguments, ensure CSV files exist
3. **Missing Target Column**: Verify target column name matches exactly (case-sensitive)
4. **Memory Issues**: Reduce `n_trials` or `cv_folds` for large datasets
5. **Slow Performance**: Use `n_jobs=-1` and reduce `max_iterations`
6. **LLM Failures**: Check API connectivity and try different models

### Debug Mode

Enable verbose output for debugging:

```python
config = AutoTunerConfig(verbose=2)
tuner = AutoTuner(config=config, ...)
```

## 📈 Performance Tips

1. **Start Small**: Begin with fewer trials and iterations, then scale up
2. **Use Parallel Processing**: Set `n_jobs=-1` for maximum speed
3. **Optimize CV Strategy**: Balance between accuracy and speed
4. **Monitor Progress**: Use progress bars to track optimization
5. **Early Stopping**: Configure `max_no_improve` to stop when converged

## 🤝 Contributing

The AutoTuner is part of the ds-assistant project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the same terms as the ds-assistant project.

---

**Happy Tuning! 🚀**
