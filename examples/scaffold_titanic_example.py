"""Complete example demonstrating GEPA scaffolding with the Titanic dataset.

This example shows how to use the scaffolding system to optimize a survival
prediction task using the classic Titanic dataset with interesting passenger features.
"""

from typing import Literal
import sys
import os

import pandas as pd
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gepa.scaffold import GepaConfig, run_optimization_pipeline
from src.gepa.data_utils import create_dataset_from_dicts
from src.gepa.lm import get_openai_model
from src.gepa.types import DataInstWithInput, RolloutOutput


# Step 1: Define input and output models
class PassengerInput(BaseModel):
    """Input features for Titanic survival prediction."""
    
    passenger_class: Literal[1, 2, 3] = Field(
        description="Passenger class (1=First, 2=Second, 3=Third)"
    )
    sex: Literal["male", "female"] = Field(
        description="Passenger's sex"
    )
    age: float = Field(
        description="Passenger's age in years"
    )
    siblings_spouses: int = Field(
        description="Number of siblings/spouses aboard"
    )
    parents_children: int = Field(
        description="Number of parents/children aboard"
    )
    fare: float = Field(
        description="Passenger fare paid in pounds"
    )
    embarked: Literal["C", "Q", "S", "Unknown"] = Field(
        description="Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)"
    )


class SurvivalPrediction(BaseModel):
    """Output prediction for Titanic survival."""
    
    survived: Literal["yes", "no"] = Field(
        description="Whether the passenger survived"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Detailed explanation of the prediction based on the features"
    )


# Step 2: Load and prepare the Titanic dataset
def load_titanic_data(n_train: int = 50, n_holdout: int = 15) -> tuple[list[dict], list[dict]]:
    """Load and prepare Titanic dataset for GEPA with holdout test set.
    
    Args:
        n_train: Number of samples to use for training/validation (default 50)
        n_holdout: Number of samples to hold out for final testing (default 15)
        
    Returns:
        Tuple of (training_data, holdout_data) as lists of dictionaries
    """
    # Load Titanic dataset from seaborn
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure seaborn is installed: pip install seaborn")
        raise
    
    # Select interesting features and clean data
    df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']].copy()
    
    # Drop rows with missing critical values
    df = df.dropna(subset=['age', 'fare'])
    
    # Fill missing embarked with 'Unknown'
    df['embarked'] = df['embarked'].fillna('Unknown')
    
    # Convert survived to string labels
    df['survived_label'] = df['survived'].map({0: 'no', 1: 'yes'})
    
    # Sample diverse examples - stratified by survival and class
    # First, separate into train and holdout sets
    train_dfs = []
    holdout_dfs = []
    
    for survived in [0, 1]:
        for pclass in [1, 2, 3]:
            subset = df[(df['survived'] == survived) & (df['pclass'] == pclass)]
            if len(subset) > 0:
                # Calculate proportional samples for this stratum
                n_train_stratum = min(len(subset), max(1, n_train // 6))
                n_holdout_stratum = min(len(subset) - n_train_stratum, max(1, n_holdout // 6))
                
                # Shuffle and split
                subset_shuffled = subset.sample(frac=1.0, random_state=42)
                train_dfs.append(subset_shuffled.iloc[:n_train_stratum])
                
                if n_holdout_stratum > 0 and len(subset_shuffled) > n_train_stratum:
                    holdout_dfs.append(subset_shuffled.iloc[n_train_stratum:n_train_stratum + n_holdout_stratum])
    
    # Combine and limit to requested sizes
    df_train = pd.concat(train_dfs, ignore_index=True)
    if len(df_train) > n_train:
        df_train = df_train.sample(n=n_train, random_state=42)
    
    df_holdout = pd.concat(holdout_dfs, ignore_index=True) if holdout_dfs else pd.DataFrame()
    if len(df_holdout) > n_holdout:
        df_holdout = df_holdout.sample(n=n_holdout, random_state=43)
    
    # Convert to list of dicts
    def df_to_dict_list(df: pd.DataFrame) -> list[dict]:
        """Convert DataFrame to list of dictionaries."""
        data = []
        for _, row in df.iterrows():
            data.append({
                'passenger_class': int(row['pclass']),
                'sex': str(row['sex']),
                'age': float(row['age']),
                'siblings_spouses': int(row['sibsp']),
                'parents_children': int(row['parch']),
                'fare': float(row['fare']),
                'embarked': str(row['embarked']),
                'label': str(row['survived_label'])
            })
        return data
    
    train_data = df_to_dict_list(df_train)
    holdout_data = df_to_dict_list(df_holdout)
    
    return train_data, holdout_data


# Step 3: Define evaluation metric
def survival_metric(
    data_inst: DataInstWithInput[PassengerInput],
    output: RolloutOutput[SurvivalPrediction],
) -> tuple[float, str | None]:
    """Evaluate survival prediction accuracy.
    
    This metric checks if the predicted survival matches the ground truth.
    It also considers confidence calibration as a bonus.
    
    Args:
        data_inst: Input data instance with metadata containing ground truth.
        output: Agent's output to evaluate.
        
    Returns:
        Tuple of (score, feedback) where score is between 0.0 and 1.0.
    """
    # Check if the agent execution was successful
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce output"
    
    # Extract predicted survival
    predicted_survival = output.result.survived
    confidence = output.result.confidence
    
    # Extract ground truth from metadata
    ground_truth = data_inst.metadata.get("label")
    
    if ground_truth is None:
        return 0.0, "No ground truth label found in metadata"
    
    # Base score: correct prediction gets 1.0, incorrect gets 0.0
    if predicted_survival == ground_truth:
        # Bonus for high confidence on correct predictions
        score = 0.7 + (0.3 * confidence)
        feedback = f"✓ Correct: {predicted_survival} (confidence: {confidence:.2f})"
    else:
        # Penalty scales with confidence on wrong predictions
        score = 0.3 * (1 - confidence)
        feedback = f"✗ Incorrect: predicted {predicted_survival}, expected {ground_truth} (confidence: {confidence:.2f})"
    
    return score, feedback


# Step 4: Main optimization pipeline
def main():
    """Run the GEPA optimization for Titanic survival prediction."""
    
    print("\n" + "="*70)
    print("Loading Titanic Dataset")
    print("="*70)
    
    # Load the data with train/holdout split
    train_data, holdout_data = load_titanic_data(n_train=50, n_holdout=5)
    
    print(f"Loaded {len(train_data)} training records")
    print(f"Loaded {len(holdout_data)} holdout test records")
    
    # Show some statistics
    train_survival_counts = {}
    for record in train_data:
        label = record['label']
        train_survival_counts[label] = train_survival_counts.get(label, 0) + 1
    
    holdout_survival_counts = {}
    for record in holdout_data:
        label = record['label']
        holdout_survival_counts[label] = holdout_survival_counts.get(label, 0) + 1
    
    print(f"Training survival distribution: {train_survival_counts}")
    print(f"Holdout survival distribution: {holdout_survival_counts}")
    
    # Convert to GEPA dataset format
    dataset = create_dataset_from_dicts(
        train_data,
        input_model=PassengerInput,
        input_keys=['passenger_class', 'sex', 'age', 'siblings_spouses', 
                   'parents_children', 'fare', 'embarked'],
        metadata_keys=['label'],
    )
    
    print(f"Created training dataset with {len(dataset)} examples")
    
    # Configure the optimization
    reflection_model = "gpt-4.1-mini"
    agent_model="gpt-4.1-mini"
    
    config = GepaConfig(
        # Agent configuration
        agent_model=agent_model,
        agent_instructions=(
            "You are an expert at predicting Titanic passenger survival. "
            "Analyze the passenger's features carefully, considering historical "
            "patterns like 'women and children first', class privileges, and "
            "family dynamics. Provide a well-reasoned prediction."
        ),
        input_type=PassengerInput,
        output_type=SurvivalPrediction,
        
        # Data and evaluation
        dataset=dataset,
        metric=survival_metric,
        train_ratio=0.7,  # 70% for training, 30% for validation
        
        # Optimization parameters
        max_metric_calls=50,  # More calls for better optimization
        module_selector="all",  # Optimize instructions, signature, and tools
        
        # LLM for reflection
        reflection_model=reflection_model,
        
        # Display options
        display_progress_bar=True,
        track_best_outputs=True,
        
        # Caching for faster iterations
        enable_cache=True,
        cache_dir=".gepa_cache",
        
        # Output settings
        output_dir="optimization_results",
        save_result=True,
    )
    
    print("\n" + "="*70)
    print("Starting GEPA Optimization")
    print("="*70)
    print("Task: Titanic Survival Prediction")
    print(f"Model: {config.agent_model}")
    print(f"Reflection Model: {config.reflection_model}")
    print(f"Dataset size: {len(config.dataset)} passengers")
    print(f"Train/Val split: {config.train_ratio:.0%} / {1-config.train_ratio:.0%}")
    print(f"Max metric calls: {config.max_metric_calls}")
    print("="*70 + "\n")
    
    # Run the optimization
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
    print(f"GEPA Input Tokens: {result.gepa_usage.input_tokens}")
    print(f"GEPA Output Tokens: {result.gepa_usage.output_tokens}")
   
    
    print("\nOptimized Components:")
    for component_name, component_value in result.best_candidate.items():
        print(f"\n{component_name}:")
        print(f"  {component_value}")
    
    print("\n" + "="*70)
    
    # Test the optimized agent on holdout set
    print("\nEvaluating optimized agent on holdout test set...")
    print("="*70)
    
    from pydantic_ai import Agent
    from src.gepa.signature_agent import SignatureAgent
    
    # Create and configure agent
    test_agent = Agent(
        model=get_openai_model(config.agent_model),
        instructions=config.agent_instructions,
        output_type=SurvivalPrediction,
    )
    
    test_signature_agent = SignatureAgent(
        test_agent,
        input_type=PassengerInput,
    )
    
    # Track results
    correct_predictions = 0
    total_predictions = 0
    results_table = []
    
    # Apply optimized configuration and test on holdout set
    with result.apply_best_to(agent=test_agent, input_type=PassengerInput):
        for i, test_record in enumerate(holdout_data, 1):
            # Create input from test record
            test_input = PassengerInput(
                passenger_class=test_record['passenger_class'],
                sex=test_record['sex'],
                age=test_record['age'],
                siblings_spouses=test_record['siblings_spouses'],
                parents_children=test_record['parents_children'],
                fare=test_record['fare'],
                embarked=test_record['embarked']
            )
            
            # Get ground truth
            actual = test_record['label']
            
            # Run prediction
            try:
                test_result = test_signature_agent.run_signature_sync(test_input)
                predicted = test_result.output.survived
                confidence = test_result.output.confidence
                reasoning = test_result.output.reasoning
                
                # Check if correct
                is_correct = predicted == actual
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store result
                results_table.append({
                    'case': i,
                    'class': test_input.passenger_class,
                    'sex': test_input.sex,
                    'age': test_input.age,
                    'predicted': predicted,
                    'actual': actual,
                    'confidence': confidence,
                    'correct': is_correct,
                    'reasoning': reasoning
                })
                
            except Exception as e:
                print(f"\n⚠️  Error on test case {i}: {e}")
                results_table.append({
                    'case': i,
                    'class': test_input.passenger_class,
                    'sex': test_input.sex,
                    'age': test_input.age,
                    'predicted': 'ERROR',
                    'actual': actual,
                    'confidence': 0.0,
                    'correct': False,
                    'reasoning': str(e)
                })
    
    # Print results table
    print(f"\nHoldout Test Results ({len(holdout_data)} passengers):")
    print("-" * 100)
    print(f"{'#':<4} {'Class':<6} {'Sex':<7} {'Age':<5} {'Predicted':<10} {'Actual':<10} {'Conf':<6} {'Result':<8}")
    print("-" * 100)
    
    for row in results_table:
        result_symbol = "✓" if row['correct'] else "✗"
        print(f"{row['case']:<4} {row['class']:<6} {row['sex']:<7} {row['age']:<5.0f} "
              f"{row['predicted']:<10} {row['actual']:<10} {row['confidence']:<6.2f} {result_symbol:<8}")
    
    print("-" * 100)
    
    # Calculate and display accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n📊 Holdout Test Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    # Show a few detailed examples
    print("\n" + "="*70)
    print("Sample Detailed Predictions:")
    print("="*70)
    
    for i, row in enumerate(results_table[:5], 1):  # Show first 5
        result_symbol = "✓ CORRECT" if row['correct'] else "✗ INCORRECT"
        print(f"\n--- Case {row['case']} ({result_symbol}) ---")
        print(f"Passenger: Class {row['class']}, {row['sex']}, age {row['age']:.0f}")
        print(f"Predicted: {row['predicted'].upper()} (confidence: {row['confidence']:.2%})")
        print(f"Actual: {row['actual'].upper()}")
        print(f"Reasoning: {row['reasoning'][:150]}..." if len(row['reasoning']) > 150 else f"Reasoning: {row['reasoning']}")
    
    print("\n" + "="*70)
    print("Example Complete!")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    # Run the example
    result = main()

