# Migration Guide: From Manual Setup to Scaffolding System

This guide helps you migrate from the manual GEPA setup to the new scaffolding system.

## Overview

The scaffolding system simplifies GEPA optimization setup by reducing boilerplate code while maintaining full flexibility. This guide shows you how to convert existing code to use the new system.

## Before and After Comparison

### Before: Manual Setup

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from src.gepa.runner import optimize_agent_prompts
from src.gepa.signature_agent import SignatureAgent
from src.gepa.types import DataInstWithInput, RolloutOutput

# Define models
class MyInput(BaseModel):
    text: str = Field(description="Input text")

class MyOutput(BaseModel):
    result: str = Field(description="Result")

# Manually create dataset
dataset = []
for item in data:
    data_inst = DataInstWithInput[MyInput](
        input=MyInput(text=item['text']),
        message_history=None,
        metadata={'label': item['label']},
        case_id=f"item-{item['id']}"
    )
    dataset.append(data_inst)

# Manually split dataset
split_idx = int(len(dataset) * 0.7)
trainset = dataset[:split_idx]
valset = dataset[split_idx:]

# Create agent
agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="Classify the text",
    output_type=MyOutput,
)

# Wrap with SignatureAgent
signature_agent = SignatureAgent(
    agent,
    input_type=MyInput,
)

# Define metric
def metric(data_inst, output):
    if not output.success or output.result is None:
        return 0.0, "Failed"
    return (1.0, "Correct") if output.result.result == data_inst.metadata['label'] else (0.0, "Incorrect")

# Run optimization
result = optimize_agent_prompts(
    agent=signature_agent,
    trainset=trainset,
    valset=valset,
    metric=metric,
    input_type=MyInput,
    max_metric_calls=100,
    display_progress_bar=True,
    track_best_outputs=True,
)

# Manually save results
import json
from datetime import datetime
from pathlib import Path

output_dir = Path("optimization_results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"optimization_{timestamp}.json"

with open(output_file, "w") as f:
    json.dump(result.model_dump(), f, indent=2)
```

### After: Scaffolding System

```python
from pydantic import BaseModel, Field
from src.gepa.scaffold import GepaConfig, setup_optimization
from src.gepa.data_utils import create_dataset_from_dicts
from src.gepa.types import DataInstWithInput, RolloutOutput

# Define models (same as before)
class MyInput(BaseModel):
    text: str = Field(description="Input text")

class MyOutput(BaseModel):
    result: str = Field(description="Result")

# Convert data using helper
dataset = create_dataset_from_dicts(
    data,
    input_type=MyInput,
    input_keys=['text'],
    metadata_keys=['label'],
    case_id_key='id'
)

# Define metric (same as before)
def metric(data_inst, output):
    if not output.success or output.result is None:
        return 0.0, "Failed"
    return (1.0, "Correct") if output.result.result == data_inst.metadata['label'] else (0.0, "Incorrect")

# Configure and run (replaces all the manual setup)
config = GepaConfig(
    agent_model="openai:gpt-4.1-mini",
    agent_instructions="Classify the text",
    input_type=MyInput,
    output_type=MyOutput,
    dataset=dataset,
    metric=metric,
    train_ratio=0.7,
    max_metric_calls=100,
)

result = setup_optimization(config)
```

**Lines of code**: ~80 lines → ~40 lines (50% reduction)

## Step-by-Step Migration

### Step 1: Keep Your Models

Your input and output models remain unchanged:

```python
# No changes needed
class MyInput(BaseModel):
    text: str = Field(description="Input text")

class MyOutput(BaseModel):
    result: str = Field(description="Result")
```

### Step 2: Convert Data Loading

#### From Manual Loop

**Before:**
```python
dataset = []
for item in data:
    data_inst = DataInstWithInput[MyInput](
        input=MyInput(text=item['text'], context=item['context']),
        message_history=None,
        metadata={'label': item['label']},
        case_id=f"item-{idx}"
    )
    dataset.append(data_inst)
```

**After:**
```python
from src.gepa.data_utils import create_dataset_from_dicts

dataset = create_dataset_from_dicts(
    data,
    input_type=MyInput,
    input_keys=['text', 'context'],
    metadata_keys=['label']
)
```

#### From DataFrame

**Before:**
```python
dataset = []
for idx, row in df.iterrows():
    data_inst = DataInstWithInput[MyInput](
        input=MyInput(text=row['text'], context=row['context']),
        message_history=None,
        metadata={'label': row['label']},
        case_id=f"row-{idx}"
    )
    dataset.append(data_inst)
```

**After:**
```python
from src.gepa.data_utils import dataframe_to_dataset

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

**Before:**
```python
import json

with open('data.json') as f:
    data = json.load(f)

dataset = []
for item in data:
    data_inst = DataInstWithInput[MyInput](
        input=MyInput(text=item['text']),
        message_history=None,
        metadata={'label': item['label']},
        case_id=item['id']
    )
    dataset.append(data_inst)
```

**After:**
```python
from src.gepa.data_utils import json_to_dataset

def dict_to_input(data):
    return MyInput(text=data['text'])

dataset = json_to_dataset(
    'data.json',
    input_type=MyInput,
    input_mapper=dict_to_input,
    metadata_keys=['label'],
    case_id_key='id'
)
```

### Step 3: Keep Your Metric

Your metric function remains unchanged:

```python
# No changes needed
def metric(data_inst, output):
    if not output.success or output.result is None:
        return 0.0, "Failed"
    # Your evaluation logic
    return score, feedback
```

### Step 4: Replace Manual Setup with Config

**Before:**
```python
# Manual train/val split
split_idx = int(len(dataset) * 0.7)
trainset = dataset[:split_idx]
valset = dataset[split_idx:]

# Create agent
agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="Your instructions",
    output_type=MyOutput,
)

# Wrap with SignatureAgent
signature_agent = SignatureAgent(
    agent,
    input_type=MyInput,
)

# Run optimization
result = optimize_agent_prompts(
    agent=signature_agent,
    trainset=trainset,
    valset=valset,
    metric=metric,
    input_type=MyInput,
    max_metric_calls=100,
    display_progress_bar=True,
    track_best_outputs=True,
)

# Manual result saving
import json
from datetime import datetime
from pathlib import Path

output_dir = Path("optimization_results")
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"optimization_{timestamp}.json"

with open(output_file, "w") as f:
    json.dump(result.model_dump(), f, indent=2)
```

**After:**
```python
from src.gepa.scaffold import GepaConfig, setup_optimization

config = GepaConfig(
    agent_model="openai:gpt-4.1-mini",
    agent_instructions="Your instructions",
    input_type=MyInput,
    output_type=MyOutput,
    dataset=dataset,
    metric=metric,
    train_ratio=0.7,
    max_metric_calls=100,
)

result = setup_optimization(config)
```

### Step 5: Using Results (No Change)

Result usage remains the same:

```python
# Access results (same as before)
print(f"Best score: {result.best_score}")
print(f"Improvement: {result.improvement_ratio()}")

# Apply to agent (same as before)
with result.apply_best_to(agent=agent, input_type=MyInput):
    output = signature_agent.run_signature_sync(test_input)
```

## Migration Checklist

- [ ] Install/update to latest version with scaffolding support
- [ ] Keep existing input/output models unchanged
- [ ] Convert data loading to use data utilities
- [ ] Keep metric function unchanged
- [ ] Replace manual setup with `GepaConfig` and `setup_optimization()`
- [ ] Test with small dataset first
- [ ] Update any scripts that parse result files (format unchanged)
- [ ] Update documentation/comments

## Advanced Features Migration

### Tools

**Before:**
```python
agent = Agent(
    model="openai:gpt-4.1-mini",
    instructions="Your instructions",
    output_type=MyOutput,
)

@agent.tool
def my_tool(query: str) -> str:
    return "result"

signature_agent = SignatureAgent(
    agent,
    input_type=MyInput,
    optimize_tools=True,
)
```

**After:**
```python
def my_tool(query: str) -> str:
    return "result"

config = GepaConfig(
    agent_model="openai:gpt-4.1-mini",
    agent_instructions="Your instructions",
    input_type=MyInput,
    output_type=MyOutput,
    dataset=dataset,
    metric=metric,
    agent_tools=[my_tool],
    optimize_tools=True,
)
```

### Caching

**Before:**
```python
from src.gepa.cache import create_cached_metric

cached_metric = create_cached_metric(
    metric,
    cache_dir=".gepa_cache",
    verbose=True
)

result = optimize_agent_prompts(
    agent=signature_agent,
    trainset=trainset,
    valset=valset,
    metric=cached_metric,
    # ...
)
```

**After:**
```python
config = GepaConfig(
    # ... other params
    metric=metric,  # Use original metric
    enable_cache=True,
    cache_dir=".gepa_cache",
    cache_verbose=True,
)
```

### Reflection Model

**Before:**
```python
from pydantic_ai.models.openai import OpenAIChatModel

reflection_model = OpenAIChatModel(model_name="gpt-4.1")

result = optimize_agent_prompts(
    agent=signature_agent,
    trainset=trainset,
    valset=valset,
    metric=metric,
    input_type=MyInput,
    reflection_model=reflection_model,
    # ...
)
```

**After:**
```python
config = GepaConfig(
    # ... other params
    reflection_model="openai:gpt-4.1",
)
```

## Benefits of Migration

1. **Less Boilerplate**: ~50% reduction in code
2. **Better Validation**: Automatic config validation with helpful errors
3. **Easier Maintenance**: Single config object vs. scattered parameters
4. **Built-in Features**: Auto-splitting, result saving, progress reporting
5. **Better Documentation**: Comprehensive docstrings and examples
6. **Type Safety**: Full type hints throughout

## Backward Compatibility

The scaffolding system is **fully backward compatible**:

- All existing code continues to work
- No breaking changes to existing APIs
- Scaffolding is an optional convenience layer
- You can mix old and new approaches

## Common Migration Issues

### Issue: Custom Train/Val Split Logic

**Solution**: Use `split_dataset()` utility or split manually before passing to config:

```python
from src.gepa.data_utils import split_dataset

# Custom split with shuffling
trainset, valset = split_dataset(
    dataset,
    train_ratio=0.8,
    shuffle=True,
    random_seed=42
)

# Then use pre-split data
# Note: Set train_ratio=1.0 to use all of dataset as training
# and provide valset separately (advanced usage)
```

### Issue: Complex Agent Setup

**Solution**: Create agent manually, then use advanced optimization:

```python
# Create complex agent manually
agent = Agent(...)
# ... complex setup ...

signature_agent = SignatureAgent(agent, input_type=MyInput)

# Use traditional optimize_agent_prompts
result = optimize_agent_prompts(
    agent=signature_agent,
    trainset=trainset,
    valset=valset,
    # ...
)
```

### Issue: Custom Result Processing

**Solution**: Scaffolding auto-saves, but you can still process results:

```python
result = setup_optimization(config)

# Custom processing
custom_process(result)

# Or disable auto-save
config = GepaConfig(
    # ...
    save_result=False,  # Disable auto-save
)
```

## Getting Help

- See `docs/SCAFFOLDING_GUIDE.md` for comprehensive documentation
- Check `examples/scaffold_example.py` for a complete working example
- Review `src/gepa/README_SCAFFOLDING.md` for quick reference
- Compare `example1.py` (manual) vs `scaffold_example.py` (scaffolding)

## Next Steps

1. Try migrating a simple example first
2. Test with a small dataset
3. Gradually migrate more complex use cases
4. Share feedback and suggestions

The scaffolding system is designed to make your life easier while maintaining full flexibility. Happy optimizing!

