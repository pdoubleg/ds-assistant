# GEPA Scaffolding System

A simplified, configuration-based interface for GEPA prompt optimization that reduces boilerplate and accelerates development.

## Quick Start

```python
from pydantic import BaseModel, Field
from src.gepa.scaffold import GepaConfig, setup_optimization
from src.gepa.data_utils import create_dataset_from_dicts

# 1. Define your models
class MyInput(BaseModel):
    text: str = Field(description="Input text")

class MyOutput(BaseModel):
    result: str = Field(description="Classification result")

# 2. Load your data
data = [
    {'text': 'Example 1', 'label': 'positive'},
    {'text': 'Example 2', 'label': 'negative'},
    # ... more examples
]

dataset = create_dataset_from_dicts(
    data,
    input_type=MyInput,
    input_keys=['text'],
    metadata_keys=['label']
)

# 3. Define your metric
def my_metric(data_inst, output):
    if not output.success or output.result is None:
        return 0.0, "Failed"
    
    predicted = output.result.result
    expected = data_inst.metadata['label']
    return (1.0, "Correct") if predicted == expected else (0.0, "Incorrect")

# 4. Configure and run
config = GepaConfig(
    agent_model="openai:gpt-4.1-mini",
    agent_instructions="Classify the input text",
    input_type=MyInput,
    output_type=MyOutput,
    dataset=dataset,
    metric=my_metric,
)

result = setup_optimization(config)
```

## Key Features

- **Configuration-driven**: Single `GepaConfig` object for all settings
- **Data utilities**: Convert DataFrames, JSON, and dicts to GEPA format
- **Template generators**: Generate boilerplate code for metrics and data loaders
- **Automatic train/val split**: Built-in dataset splitting
- **Result persistence**: Auto-save optimization results to JSON

## Components

### Core
- `GepaConfig`: Configuration dataclass for all optimization parameters
- `setup_optimization()`: Main function to run optimization

### Data Utilities
- `dataframe_to_dataset()`: Convert pandas DataFrames
- `json_to_dataset()`: Load from JSON files
- `create_dataset_from_dicts()`: Convert list of dicts
- `split_dataset()`: Split data into train/val sets

### Templates
- `generate_metric_template()`: Generate metric function code
- `generate_data_loader_template()`: Generate data loading code
- `generate_example_script()`: Generate complete example scripts
- `print_metric_template()`: Print templates to console
- `print_data_loader_template()`: Print data loader templates

## Examples

### From DataFrame
```python
import pandas as pd
from src.gepa.data_utils import dataframe_to_dataset

df = pd.read_csv("data.csv")

def row_to_input(row):
    return MyInput(text=row['text'])

dataset = dataframe_to_dataset(
    df,
    input_type=MyInput,
    row_mapper=row_to_input,
    metadata_cols=['label']
)
```

### From JSON
```python
from src.gepa.data_utils import json_to_dataset

def dict_to_input(data):
    return MyInput(text=data['text'])

dataset = json_to_dataset(
    'data.json',
    input_type=MyInput,
    input_mapper=dict_to_input,
    metadata_keys=['label']
)
```

### Generate Templates
```python
from src.gepa.templates import print_metric_template, print_data_loader_template

# Print metric template
print_metric_template("exact_match")

# Print data loader template
print_data_loader_template("dataframe")
```

## Configuration Options

### Essential Parameters
- `agent_model`: Model name (e.g., "openai:gpt-4.1-mini")
- `agent_instructions`: System instructions
- `input_type`: Pydantic model for inputs
- `output_type`: Pydantic model for outputs
- `dataset`: List of DataInstWithInput instances
- `metric`: Evaluation function

### Optimization Parameters
- `train_ratio`: Train/val split ratio (default: 0.7)
- `max_metric_calls`: Max evaluations (default: 100)
- `module_selector`: Components to optimize (default: "all")
- `reflection_model`: Model for reflection (default: None)

### Tool Configuration
- `agent_tools`: List of tool functions (default: None)
- `optimize_tools`: Optimize tool descriptions (default: True)

### Runtime Options
- `display_progress_bar`: Show progress (default: True)
- `enable_cache`: Cache results (default: False)
- `save_result`: Auto-save to JSON (default: True)
- `output_dir`: Results directory (default: "optimization_results")

## Complete Example

See `examples/scaffold_example.py` for a complete working example:

```bash
python examples/scaffold_example.py
```

## Documentation

For detailed documentation, see:
- [Scaffolding Guide](../../docs/SCAFFOLDING_GUIDE.md) - Complete guide with examples
- [Main GEPA Documentation](../../README.md) - Full GEPA documentation

## Metric Patterns

### Exact Match
```python
def exact_match_metric(data_inst, output):
    if not output.success or output.result is None:
        return 0.0, "Failed"
    
    predicted = output.result.category
    expected = data_inst.metadata['label']
    
    return (1.0, "Correct") if predicted == expected else (0.0, "Incorrect")
```

### LLM Judge
```python
def llm_judge_metric(data_inst, output):
    if not output.success or output.result is None:
        return 0.0, "Failed"
    
    # Create judge agent
    judge = Agent(
        model="openai:gpt-4.1-mini",
        instructions="Evaluate the output quality",
        output_type=EvaluationOutput
    )
    
    # Run evaluation
    eval_result = judge.run_sync(
        f"Input: {data_inst.input}\nOutput: {output.result}"
    )
    
    return eval_result.output.score, eval_result.output.feedback
```

### Custom Scoring
```python
def custom_metric(data_inst, output):
    if not output.success or output.result is None:
        return 0.0, "Failed"
    
    # Custom logic
    score = calculate_similarity(output.result, data_inst.metadata['expected'])
    feedback = f"Similarity: {score:.2f}"
    
    return score, feedback
```

## Best Practices

1. **Start with small datasets** (10-20 examples) for testing
2. **Enable caching** during development for faster iteration
3. **Use appropriate train_ratio** (0.7-0.8 for most cases)
4. **Set reasonable max_metric_calls** to control costs
5. **Test metrics independently** before optimization
6. **Save results** for tracking optimization history
7. **Use stronger reflection models** for better results

## Troubleshooting

**Empty dataset**: Ensure at least 2 examples for train/val split

**Invalid train_ratio**: Adjust ratio or add more data

**Missing metadata**: Check metadata_cols/metadata_keys match your data

**Metric errors**: Ensure metric returns `tuple[float, str | None]`

## API Reference

### GepaConfig
```python
@dataclass
class GepaConfig:
    agent_model: Model | KnownModelName | str
    agent_instructions: str
    input_type: type[BaseModel]
    output_type: type[BaseModel]
    dataset: Sequence[DataInstWithInput[Any]]
    metric: Callable[[DataInstWithInput[Any], RolloutOutput[Any]], tuple[float, str | None]]
    train_ratio: float = 0.7
    # ... more parameters
```

### setup_optimization
```python
def setup_optimization(config: GepaConfig) -> GepaOptimizationResult:
    """Run GEPA optimization based on configuration."""
```

### Data Utilities
```python
def dataframe_to_dataset(
    df: Any,
    input_type: type[InputModelT],
    row_mapper: Callable[[Any], InputModelT],
    metadata_cols: list[str] | None = None,
    case_id_col: str | None = None,
) -> list[DataInstWithInput[InputModelT]]:
    """Convert DataFrame to dataset."""

def json_to_dataset(
    json_path: str | Path,
    input_type: type[InputModelT],
    input_mapper: Callable[[dict[str, Any]], InputModelT],
    metadata_keys: list[str] | None = None,
    case_id_key: str | None = None,
) -> list[DataInstWithInput[InputModelT]]:
    """Load dataset from JSON."""

def create_dataset_from_dicts(
    data: list[dict[str, Any]],
    input_type: type[InputModelT],
    input_keys: list[str],
    metadata_keys: list[str] | None = None,
    case_id_key: str | None = None,
) -> list[DataInstWithInput[InputModelT]]:
    """Create dataset from list of dicts."""
```

## License

Same as the main GEPA project.

