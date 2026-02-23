# GEPA Scaffolding System Guide

The GEPA scaffolding system provides a simplified, configuration-based approach to setting up prompt optimization for your pydantic-ai agents. This guide will help you get started quickly.

## Overview

The scaffolding system reduces the boilerplate needed to run GEPA optimization by providing:

1. **Configuration-based setup** - Define all parameters in a single `GepaConfig` object
2. **Data loading utilities** - Helper functions to convert DataFrames, JSON, and dicts to GEPA format
3. **Template generators** - Code templates for common patterns (metrics, data loaders)
4. **Simplified workflow** - One function call to run complete optimization

## Quick Start

### 1. Define Your Models

```python
from pydantic import BaseModel, Field

class MyInput(BaseModel):
    """Your input structure."""
    text: str = Field(description="Input text")
    context: str = Field(description="Additional context")

class MyOutput(BaseModel):
    """Your output structure."""
    result: str = Field(description="The result")
    confidence: float = Field(description="Confidence score")
```

### 2. Load Your Data

Choose the method that fits your data source:

#### From DataFrame
```python
import pandas as pd
from src.gepa.data_utils import dataframe_to_dataset

df = pd.read_csv("data.csv")

def row_to_input(row):
    return MyInput(text=row['text'], context=row['context'])

dataset = dataframe_to_dataset(
    df,
    input_type=MyInput,
    row_mapper=row_to_input,
    metadata_cols=['label']
)
```

#### From JSON
```python
from src.gepa.data_utils import json_to_dataset

def dict_to_input(data):
    return MyInput(text=data['text'], context=data['context'])

dataset = json_to_dataset(
    'data.json',
    input_type=MyInput,
    input_mapper=dict_to_input,
    metadata_keys=['label']
)
```

#### From List of Dicts
```python
from src.gepa.data_utils import create_dataset_from_dicts

data = [
    {'text': 'Example 1', 'context': 'Context 1', 'label': 'positive'},
    {'text': 'Example 2', 'context': 'Context 2', 'label': 'negative'},
]

dataset = create_dataset_from_dicts(
    data,
    input_type=MyInput,
    input_keys=['text', 'context'],
    metadata_keys=['label']
)
```

### 3. Define Your Metric

```python
from src.gepa.types import DataInstWithInput, RolloutOutput

def my_metric(
    data_inst: DataInstWithInput[MyInput],
    output: RolloutOutput[MyOutput],
) -> tuple[float, str | None]:
    """Evaluate the agent's output."""
    if not output.success or output.result is None:
        return 0.0, "Failed to produce output"
    
    # Your evaluation logic here
    predicted = output.result.result
    expected = data_inst.metadata.get("label")
    
    if predicted == expected:
        return 1.0, "Correct"
    else:
        return 0.0, f"Incorrect: {predicted} vs {expected}"
```

### 4. Configure and Run

```python
from src.gepa.scaffold import GepaConfig, setup_optimization

config = GepaConfig(
    # Agent setup
    agent_model="openai:gpt-4.1-mini",
    agent_instructions="Your task description here",
    input_type=MyInput,
    output_type=MyOutput,
    
    # Data and evaluation
    dataset=dataset,
    metric=my_metric,
    train_ratio=0.7,
    
    # Optimization settings
    max_metric_calls=100,
)

result = setup_optimization(config)
```

## Configuration Options

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `agent_model` | `str` | Model name (e.g., "openai:gpt-4.1-mini") | Required |
| `agent_instructions` | `str` | System instructions for the agent | Required |
| `input_type` | `type[BaseModel]` | Pydantic model for inputs | Required |
| `output_type` | `type[BaseModel]` | Pydantic model for outputs | Required |
| `dataset` | `list[DataInstWithInput]` | Training/validation data | Required |
| `metric` | `Callable` | Evaluation function | Required |

### Optimization Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `train_ratio` | `float` | Ratio for train/val split | `0.7` |
| `max_metric_calls` | `int` | Maximum evaluations | `100` |
| `module_selector` | `str` | Components to optimize ("all", "instructions", etc.) | `"all"` |
| `reflection_model` | `str` | Model for reflection/mutation | `None` (uses agent_model) |

### Tool Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `agent_tools` | `list[Callable]` | Tool functions for the agent | `None` |
| `optimize_tools` | `bool` | Whether to optimize tool descriptions | `True` |

### Runtime Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `display_progress_bar` | `bool` | Show progress during optimization | `True` |
| `track_best_outputs` | `bool` | Track best outputs for analysis | `True` |
| `enable_cache` | `bool` | Cache metric results | `False` |
| `cache_dir` | `str` | Directory for cache | `".gepa_cache"` |
| `save_result` | `bool` | Auto-save results to JSON | `True` |
| `output_dir` | `str` | Directory for results | `"optimization_results"` |

## Template Generators

The scaffolding system includes template generators to help you get started:

### Generate Metric Templates

```python
from src.gepa.templates import generate_metric_template, print_metric_template

# Generate exact match metric
template = generate_metric_template(MyInput, MyOutput, "exact_match")
print(template)

# Generate LLM judge metric
template = generate_metric_template(MyInput, MyOutput, "llm_judge")
print(template)

# Or print directly to console
print_metric_template("exact_match")
```

### Generate Data Loader Templates

```python
from src.gepa.templates import generate_data_loader_template, print_data_loader_template

# Generate DataFrame loader
template = generate_data_loader_template("dataframe")
print(template)

# Generate JSON loader
template = generate_data_loader_template("json")
print(template)

# Or print directly to console
print_data_loader_template("dataframe")
```

### Generate Complete Example Script

```python
from src.gepa.templates import generate_example_script

# Generate a complete example
script = generate_example_script("sentiment analysis", include_tools=False)
print(script)

# Save to file
with open("my_optimization.py", "w") as f:
    f.write(script)
```

## Data Utilities

### Split Dataset

```python
from src.gepa.data_utils import split_dataset

trainset, valset = split_dataset(
    dataset,
    train_ratio=0.8,
    shuffle=True,
    random_seed=42
)
```

### DataFrame to Dataset

```python
from src.gepa.data_utils import dataframe_to_dataset

dataset = dataframe_to_dataset(
    df,
    input_type=MyInput,
    row_mapper=row_to_input,
    metadata_cols=['label', 'category'],
    case_id_col='id'  # Optional: use specific column for IDs
)
```

### JSON to Dataset

```python
from src.gepa.data_utils import json_to_dataset

dataset = json_to_dataset(
    'data.json',
    input_type=MyInput,
    input_mapper=dict_to_input,
    metadata_keys=['label'],
    case_id_key='id'  # Optional: use specific key for IDs
)
```

### Dict List to Dataset

```python
from src.gepa.data_utils import create_dataset_from_dicts

dataset = create_dataset_from_dicts(
    data,
    input_type=MyInput,
    input_keys=['text', 'context'],
    metadata_keys=['label'],
    case_id_key='id'  # Optional
)
```

## Using Optimization Results

After optimization completes, you can use the results in several ways:

### Apply to Agent

```python
from pydantic_ai import Agent
from src.gepa.signature_agent import SignatureAgent

# Create your agent
agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="Original instructions",
    output_type=MyOutput
)

signature_agent = SignatureAgent(agent, input_type=MyInput)

# Apply optimized prompts
with result.apply_best_to(agent=agent, input_type=MyInput):
    output = signature_agent.run_signature_sync(test_input)
```

### Access Results

```python
# Best score
print(f"Best score: {result.best_score}")

# Original score (if evaluated)
print(f"Original score: {result.original_score}")

# Improvement ratio
improvement = result.improvement_ratio()
if improvement:
    print(f"Improvement: {improvement:.2%}")

# Optimized components
for name, value in result.best_candidate.items():
    print(f"{name}: {value}")

# Metadata
print(f"Iterations: {result.num_iterations}")
print(f"Metric calls: {result.num_metric_calls}")
```

## Complete Example

See `examples/scaffold_example.py` for a complete working example that demonstrates:
- Defining input/output models
- Loading data from a list of dicts
- Creating a simple exact-match metric
- Configuring and running optimization
- Using the optimized agent

Run it with:
```bash
python examples/scaffold_example.py
```

## Best Practices

1. **Start Small**: Begin with a small dataset (10-20 examples) to test your setup
2. **Use Caching**: Enable caching for faster iteration during development
3. **Monitor Costs**: Set `max_metric_calls` to control optimization costs
4. **Validate Metrics**: Test your metric function independently before optimization
5. **Save Results**: Keep `save_result=True` to track optimization history
6. **Use Strong Reflection Models**: Better reflection models lead to better optimizations

## Troubleshooting

### Empty Dataset Error
```
ValueError: Dataset must contain at least 2 examples for train/val split
```
**Solution**: Ensure your dataset has at least 2 examples. For production, use 10+ examples.

### Invalid Train Ratio
```
ValueError: Invalid train_ratio 0.9 for dataset size 5. Results in empty train or validation set.
```
**Solution**: Adjust `train_ratio` or add more data. With 5 examples, use `train_ratio=0.6` or `0.8`.

### Missing Metadata
```
KeyError: 'label'
```
**Solution**: Ensure your metadata_cols/metadata_keys match your data structure.

### Metric Returns Wrong Type
```
TypeError: metric must return tuple[float, str | None]
```
**Solution**: Ensure your metric function returns `(score, feedback)` where score is a float.

## Next Steps

- Review the [main GEPA documentation](../README.md) for advanced features
- Check out `example1.py` and `example2.py` for more complex use cases
- Explore the `templates.py` module for additional code generation options
- Join the community to share your optimization results!

