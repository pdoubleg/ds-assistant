

# LLM DS Assistant

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management. From the repo root:

```bash
uv sync
```

Subprojects (e.g. `agui_v2/agent`, `agui_v3/agent`) have their own `pyproject.toml`; run `uv sync` in those directories for isolated installs, or use `uv run` to execute scripts with the correct environment.

# 🧠 Architecture

```
sql
                         +------------------------+
                         |    Tabular Dataset     |
                         +-----------+------------+
                                     |
              +----------------------|----------------------+
              |                                              |
     +--------v--------+                            +--------v--------+
     |   CAAFE Agent   |                            |   Tuning Agent   |
     |  (Feature LLM)  |                            |  (Search LLM)    |
     +--------+--------+                            +--------+--------+
              |                                              |
    +---------v----------+                       +-----------v-----------+
    | New Feature Code   |                       | Suggested Search Space |
    +---------+----------+                       +-----------+-----------+
              |                                              |
    +---------v----------+                       +-----------v-----------+
    | CV-based Evaluator |                       | HalvingRandomSearchCV |
    +---------+----------+                       +-----------+-----------+
              |                                              |
     +--------v--------+                            +--------v--------+
     | Final Feature   |                            | Best Configs    |
     | Set (Transform) |                            | + Score History |
     +-----------------+                            +-----------------+

```


# CAAFE: Context-Aware Automated Feature Engineering with LLMs

Inspired by the paper [here](https://arxiv.org/pdf/2305.03403).

> ⚙️ Automatically generate meaningful features using Large Language Models + statistical validation
> 📊 Built for tabular classification tasks (esp. binary classification)
> 🧠 Designed by data scientists, for data scientists

---

## 🚀 Overview

**CAAFE (Context-Aware Automated Feature Engineering)** is a Python module that uses **LLMs** (like OpenAI's GPT-4) to generate **semantically meaningful, statistically validated features** for structured (tabular) datasets.

It tightly integrates LLM-driven code generation with fold-based evaluation to ensure that only performance-improving features are retained. The system is safe, extensible, and ideal for datasets where domain knowledge is hard to encode manually.

---

## ✨ Key Features

- 🔁 **Iterative Feature Engineering**: Multiple LLM-driven iterations with early stopping.
- 📈 **Cross-Validated Evaluation**: Accept features only if they improve metrics (AUC or Accuracy).
- 🧠 **Contextual Prompting**: LLM has access to dataset summaries, feature stats, mutual information, and correlation tools.
- 🔐 **Safe Code Validation**: All generated Python code is validated using AST parsing and a custom whitelist.
- 📜 **Explainable & Reproducible**: Each feature is saved with its reasoning and can be exported to `.py` or `.md`.
- 🧰 **Compatible with sklearn Pipelines**: Provided as a drop-in `sklearn.TransformerMixin`.



| Area            | Pattern Used                            | Notes                               |
| --------------- | --------------------------------------- | ----------------------------------- |
| Validation      | Pydantic + AST check                    | Ensures safe, interpretable code    |
| Modularity      | Sklearn Transformer, Tools, Agent       | Each component is loosely coupled   |
| Logging         | Structured + Human-readable             | Supports debugging, reproducibility |
| Evaluation      | CV-based wrapper                        | Metric-agnostic, robust             |
| LLM integration | Tool-augmented, prompt-reflective       | Avoids static prompts               |
| Persistence     | `.save_code()`and `.load_code_path` | Enables reuse in deployment         |

---

```
python
from src import CAAFETransformer
from sklearn.ensemble import RandomForestClassifier

# Assume df contains your tabular dataset and 'target' is your target column
X = df.drop(columns="target")
y = df["target"]

transformer = CAAFETransformer(
    target_name="target",
    dataset_description="Binary classification of customer churn",
    optimization_metric="auc",
    base_classifier=RandomForestClassifier(),
    llm_model="gpt-4",
    iterations=5
)

# Run the iterative feature engineering
transformer.fit(X, y)

# Apply to training or test data
X_enhanced = transformer.transform(X)

# Save the generated features for reuse
transformer.save_code("generated_features.md") # or as .py
```

```

```

### 🧪 2. LLM-Based Hyperparameter Tuning

**XGBoostTuner** is a smart AutoML component that guides the search for optimal model settings using LLMs and search history.

* 📊 Uses `HalvingRandomSearchCV` to efficiently search space
* 🔄 Automatically refines the search space over multiple rounds
* 🧠 LLM explains each suggestion using dataset characteristics and best practices
* 🔍 Tracks best configurations, iterations, and score progression
* 📈 Supports any sklearn scoring metric (e.g., `f1`, `roc_auc`, `precision`)


```
python
from src import XGBoostTuner

tuner = XGBoostTuner(dataset=df, target="target")
tuner.tune(max_iterations=5)

best_config = tuner.get_best_config()
print("Best XGBoost config:", best_config)
```

```

```
