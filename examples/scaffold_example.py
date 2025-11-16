"""Complete example demonstrating the GEPA scaffolding system.

This example shows how to use the scaffolding system to quickly set up
GEPA optimization for a sentiment classification task with minimal boilerplate.
"""

from typing import Literal

from pydantic import BaseModel, Field

import sys
import os

from pydantic_ai.models.openai import OpenAIChatModel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gepa.scaffold import GepaConfig, run_optimization_pipeline
from src.gepa.data_utils import prepare_train_val_sets
from src.gepa.types import DataInstWithInput, RolloutOutput


# Step 1: Define your input and output models
class SentimentInput(BaseModel):
    """Input for sentiment classification."""
    
    text: str = Field(description="The text to classify")
    context: str = Field(description="The context of the text")


class SentimentOutput(BaseModel):
    """Output for sentiment classification."""
    
    category: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment category"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification"
    )


# Step 2: Prepare your data
# In a real scenario, you might load this from a CSV, JSON, or database
sample_data = [
    {
        "text": "This product exceeded all my expectations!",
        "context": "Product review",
        "label": "positive",
    },
    {
        "text": "Terrible service, would not recommend.",
        "context": "Service feedback",
        "label": "negative",
    },
    {
        "text": "It's okay, nothing special.",
        "context": "General feedback",
        "label": "neutral",
    },
    {
        "text": "Absolutely love it! Best purchase ever.",
        "context": "Product review",
        "label": "positive",
    },
    {
        "text": "Disappointed with the quality.",
        "context": "Product review",
        "label": "negative",
    },
    {
        "text": "It works as described.",
        "context": "Product review",
        "label": "neutral",
    },
    {
        "text": "Outstanding customer support!",
        "context": "Service feedback",
        "label": "positive",
    },
    {
        "text": "Complete waste of money.",
        "context": "Product review",
        "label": "negative",
    },
    {
        "text": "Average experience, nothing remarkable.",
        "context": "General feedback",
        "label": "neutral",
    },
    {
        "text": "Highly recommend to everyone!",
        "context": "Product review",
        "label": "positive",
    },
    {
        "text": "Very poor quality control.",
        "context": "Product review",
        "label": "negative",
    },
    {
        "text": "It's fine for the price.",
        "context": "Product review",
        "label": "neutral",
    },
    {
        "text": "Exceeded my expectations in every way!",
        "context": "Service feedback",
        "label": "positive",
    },
    {
        "text": "Worst experience I've had.",
        "context": "Service feedback",
        "label": "negative",
    },
    {
        "text": "Does what it's supposed to do.",
        "context": "Product review",
        "label": "neutral",
    },
]

# Convert data to GEPA format and split into train/val using the helper function
trainset, valset = prepare_train_val_sets(
    sample_data,
    input_model=SentimentInput,
    input_keys=["text", "context"],
    metadata_keys=["label"],
    train_ratio=0.7,
    shuffle=True,
    random_seed=42,
)

print(f"Loaded {len(trainset)} training and {len(valset)} validation examples")


# Step 3: Define your evaluation metric
def sentiment_metric(
    data_inst: DataInstWithInput[SentimentInput],
    output: RolloutOutput[SentimentOutput],
) -> tuple[float, str | None]:
    """Evaluate sentiment classification output.
    
    This metric checks if the predicted category matches the ground truth label.
    
    Args:
        data_inst: Input data instance with metadata containing the ground truth label.
        output: Agent's output to evaluate.
        
    Returns:
        Tuple of (score, feedback) where score is 1.0 for correct, 0.0 for incorrect.
    """
    # Check if the agent execution was successful
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce output"
    
    # Extract predicted category
    predicted_category = output.result.category
    
    # Extract ground truth from metadata
    ground_truth = data_inst.metadata.get("label")
    
    if ground_truth is None:
        return 0.0, "No ground truth label found in metadata"
    
    # Compare prediction with ground truth
    if predicted_category == ground_truth:
        return 1.0, f"Correct classification: {predicted_category}"
    else:
        return 0.0, f"Incorrect: predicted {predicted_category}, expected {ground_truth}"


# Step 4: Configure and run optimization
def main():
    """Run the GEPA optimization with scaffolding."""
    
    
    reflection_model = OpenAIChatModel(
        model_name="gpt-4.1",
    )
    # Create configuration
    config = GepaConfig(
        # Agent configuration
        agent_model="openai:gpt-4.1-mini",
        agent_instructions="Classify the sentiment of the given text as positive, negative, or neutral.",
        input_type=SentimentInput,
        output_type=SentimentOutput,
        
        # Data and evaluation
        trainset=trainset,
        valset=valset,
        metric=sentiment_metric,
        
        # Optimization parameters
        max_metric_calls=50,  # Limit for demonstration purposes
        module_selector="all",  # Optimize all components (instructions, signature, tools)
        
        # Optional: Use a more powerful model for reflection
        reflection_model=reflection_model,
        
        # Display options
        display_progress_bar=True,
        track_best_outputs=True,
        
        # Caching (optional, speeds up repeated runs)
        enable_cache=True,
        cache_dir=".gepa_cache",
        
        # Output settings
        output_dir="optimization_results",
        save_result=True,
    )
    
    print("\n" + "="*70)
    print("Starting GEPA Optimization with Scaffolding")
    print("="*70)
    print("Task: Sentiment Classification")
    print(f"Model: {config.agent_model}")
    print(f"Training set: {len(config.trainset)} examples")
    print(f"Validation set: {len(config.valset) if config.valset else 0} examples")
    print(f"Max metric calls: {config.max_metric_calls}")
    print("="*70 + "\n")
    
    # Run optimization using the scaffolding system
    result = run_optimization_pipeline(config)
    
    # Display results
    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)
    print(f"Best Score: {result.best_score:.4f}")
    
    if result.original_score is not None:
        print(f"Original Score: {result.original_score:.4f}")
        improvement = result.improvement_ratio()
        if improvement is not None:
            print(f"Improvement: {improvement:+.2%}")
    
    print(f"Iterations: {result.num_iterations}")
    print(f"Metric Calls: {result.num_metric_calls}")
    
    print("\nOptimized Components:")
    for component_name, component_value in result.best_candidate.items():
        print(f"\n{component_name}:")
        # Truncate long values for display
        if len(component_value) > 200:
            print(f"  {component_value[:200]}...")
        else:
            print(f"  {component_value}")
    
    print("\n" + "="*70)
    
    # Demonstrate using the optimized agent
    print("\nTesting optimized agent on a new example...")
    
    from pydantic_ai import Agent
    from src.gepa.signature_agent import SignatureAgent
    
    # Create a fresh agent
    test_agent = Agent(
        model=config.agent_model,
        instructions=config.agent_instructions,
        output_type=SentimentOutput,
    )
    
    # Wrap with SignatureAgent
    test_signature_agent = SignatureAgent(
        test_agent,
        input_type=SentimentInput,
    )
    
    # Apply the best candidate
    with result.apply_best_to(agent=test_agent, input_type=SentimentInput):
        test_input = SentimentInput(
            text="This is amazing! I'm so happy with it!",
            context="Product review"
        )
        
        test_result = test_signature_agent.run_signature_sync(test_input)
        
        print(f"\nTest Input: {test_input.text}")
        print(f"Predicted Category: {test_result.output.category}")
        print(f"Confidence: {test_result.output.confidence:.2f}")
        print(f"Reasoning: {test_result.output.reasoning}")
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    # Run the example
    result = main()
    
