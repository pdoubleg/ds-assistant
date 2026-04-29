"""Complete example demonstrating GEPA scaffolding with the Iris dataset.

This example shows how to use the scaffolding system to optimize a flower species
classification task using the classic Iris dataset with sepal and petal measurements.
"""

from typing import Literal
import sys
import os

import pandas as pd
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gepa.scaffold import GepaConfig, run_optimization_pipeline
from src.gepa.data_utils import prepare_train_val_sets
from src.gepa.types import DataInstWithInput, RolloutOutput


# Step 1: Define input and output models
class IrisInput(BaseModel):
    """Input features for Iris flower classification."""

    sepal_length: float = Field(description="Sepal length in centimeters")
    sepal_width: float = Field(description="Sepal width in centimeters")
    petal_length: float = Field(description="Petal length in centimeters")
    petal_width: float = Field(description="Petal width in centimeters")


class IrisClassification(BaseModel):
    """Output prediction for Iris flower species."""

    species: Literal["setosa", "versicolor", "virginica"] = Field(
        description="The predicted Iris species"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Detailed explanation of the classification based on the measurements"
    )


# Step 2: Load and prepare the Iris dataset
def load_iris_data(
    n_train: int = 60, n_holdout: int = 15
) -> tuple[list[dict], list[dict]]:
    """Load and prepare Iris dataset for GEPA with holdout test set.

    Args:
        n_train: Number of samples to use for training/validation (default 60)
        n_holdout: Number of samples to hold out for final testing (default 15)

    Returns:
        Tuple of (training_data, holdout_data) as lists of dictionaries

    Example:
        >>> train_data, holdout_data = load_iris_data(n_train=60, n_holdout=15)
        >>> print(f"Training samples: {len(train_data)}")
        Training samples: 60
    """
    # Load Iris dataset from sklearn
    try:
        from sklearn.datasets import load_iris

        iris = load_iris(as_frame=True)
        df = iris.frame
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure scikit-learn is installed: pip install scikit-learn")
        raise

    # Rename columns for clarity
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    # Map target to species names
    species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    df["species"] = df["target"].map(species_map)

    # Sample diverse examples - stratified by species
    # First, separate into train and holdout sets
    train_dfs = []
    holdout_dfs = []

    for species_id in [0, 1, 2]:
        subset = df[df["target"] == species_id]
        if len(subset) > 0:
            # Calculate proportional samples for this stratum
            n_train_stratum = min(len(subset), max(1, n_train // 3))
            n_holdout_stratum = min(
                len(subset) - n_train_stratum, max(1, n_holdout // 3)
            )

            # Shuffle and split
            subset_shuffled = subset.sample(frac=1.0, random_state=42)
            train_dfs.append(subset_shuffled.iloc[:n_train_stratum])

            if n_holdout_stratum > 0 and len(subset_shuffled) > n_train_stratum:
                holdout_dfs.append(
                    subset_shuffled.iloc[
                        n_train_stratum : n_train_stratum + n_holdout_stratum
                    ]
                )

    # Combine and limit to requested sizes
    df_train = pd.concat(train_dfs, ignore_index=True)
    if len(df_train) > n_train:
        df_train = df_train.sample(n=n_train, random_state=42)

    df_holdout = (
        pd.concat(holdout_dfs, ignore_index=True) if holdout_dfs else pd.DataFrame()
    )
    if len(df_holdout) > n_holdout:
        df_holdout = df_holdout.sample(n=n_holdout, random_state=43)

    # Convert to list of dicts
    def df_to_dict_list(df: pd.DataFrame) -> list[dict]:
        """Convert DataFrame to list of dictionaries.

        Args:
            df: DataFrame containing Iris data

        Returns:
            List of dictionaries with features and labels
        """
        data = []
        for _, row in df.iterrows():
            data.append(
                {
                    "sepal_length": float(row["sepal_length"]),
                    "sepal_width": float(row["sepal_width"]),
                    "petal_length": float(row["petal_length"]),
                    "petal_width": float(row["petal_width"]),
                    "label": str(row["species"]),
                }
            )
        return data

    train_data = df_to_dict_list(df_train)
    holdout_data = df_to_dict_list(df_holdout)

    return train_data, holdout_data


# Step 3: Define evaluation metric
def iris_metric(
    data_inst: DataInstWithInput[IrisInput],
    output: RolloutOutput[IrisClassification],
) -> tuple[float, str | None]:
    """Evaluate Iris classification accuracy.

    This metric checks if the predicted species matches the ground truth.
    It also considers confidence calibration as a bonus.

    Args:
        data_inst: Input data instance with metadata containing ground truth.
        output: Agent's output to evaluate.

    Returns:
        Tuple of (score, feedback) where score is between 0.0 and 1.0.

    Example:
        >>> # Assuming data_inst and output are properly constructed
        >>> score, feedback = iris_metric(data_inst, output)
        >>> assert 0.0 <= score <= 1.0
    """
    # Check if the agent execution was successful
    if not output.success or output.result is None:
        return 0.0, output.error_message or "Agent failed to produce output"

    # Extract predicted species
    predicted_species = output.result.species
    confidence = output.result.confidence

    # Extract ground truth from metadata
    ground_truth = data_inst.metadata.get("label")

    if ground_truth is None:
        return 0.0, "No ground truth label found in metadata"

    # Base score: correct prediction gets 1.0, incorrect gets 0.0
    if predicted_species == ground_truth:
        # Bonus for high confidence on correct predictions
        score = 0.7 + (0.3 * confidence)
        feedback = f"✓ Correct: {predicted_species} (confidence: {confidence:.2f})"
    else:
        # Penalty scales with confidence on wrong predictions
        score = 0.3 * (1 - confidence)
        feedback = f"✗ Incorrect: predicted {predicted_species}, expected {ground_truth} (confidence: {confidence:.2f})"

    return score, feedback


# Step 4: Main optimization pipeline
def main():
    """Run the GEPA optimization for Iris flower classification.

    This function orchestrates the entire optimization process:
    1. Loads the Iris dataset
    2. Configures the GEPA optimization
    3. Runs the optimization pipeline
    4. Evaluates on a holdout test set
    5. Displays comprehensive results

    Returns:
        OptimizationResult: The result object containing the best configuration
    """

    print("\n" + "=" * 70)
    print("Loading Iris Dataset")
    print("=" * 70)

    # Load the data with train/holdout split
    train_data, holdout_data = load_iris_data(n_train=60, n_holdout=15)

    print(f"Loaded {len(train_data)} training records")
    print(f"Loaded {len(holdout_data)} holdout test records")

    # Show some statistics
    train_species_counts = {}
    for record in train_data:
        label = record["label"]
        train_species_counts[label] = train_species_counts.get(label, 0) + 1

    holdout_species_counts = {}
    for record in holdout_data:
        label = record["label"]
        holdout_species_counts[label] = holdout_species_counts.get(label, 0) + 1

    print(f"Training species distribution: {train_species_counts}")
    print(f"Holdout species distribution: {holdout_species_counts}")

    # Convert to GEPA dataset format and split into train/val
    trainset, valset = prepare_train_val_sets(
        train_data,
        input_model=IrisInput,
        input_keys=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        metadata_keys=["label"],
        train_ratio=0.7,
        shuffle=True,
        random_seed=42,
    )

    print(
        f"Created dataset: {len(trainset)} training, {len(valset)} validation examples"
    )

    # Configure the optimization
    reflection_model = "gpt-4.1-mini"
    agent_model = "gpt-4.1-mini"

    config = GepaConfig(
        # Agent configuration
        agent_model=agent_model,
        agent_instructions=(
            "You are an expert botanist specializing in Iris flower classification. "
            "Analyze the sepal and petal measurements carefully to determine the species. "
            "Consider that setosa typically has smaller petals, while virginica has the "
            "largest measurements overall. Versicolor falls in between. Provide a "
            "well-reasoned classification based on the morphological features."
        ),
        input_type=IrisInput,
        output_type=IrisClassification,
        # Data and evaluation
        trainset=trainset,
        valset=valset,
        metric=iris_metric,
        # Optimization parameters
        max_metric_calls=100,  # More calls for better optimization
        module_selector="round_robin",  # Optimize instructions, signature, and tools
        # Merge options
        use_merge=False,
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

    print("\n" + "=" * 70)
    print("Starting GEPA Optimization")
    print("=" * 70)
    print("Task: Iris Flower Classification")
    print(f"Model: {config.agent_model}")
    print(f"Reflection Model: {reflection_model}")
    print(f"Training set: {len(config.trainset)} flowers")
    print(f"Validation set: {len(config.valset) if config.valset else 0} flowers")
    print(f"Max metric calls: {config.max_metric_calls}")
    print("=" * 70 + "\n")

    # Run the optimization
    result = run_optimization_pipeline(config)

    # Display results
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
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
        # Truncate long values for display
        if isinstance(component_value, str) and len(component_value) > 300:
            print(f"  {component_value[:300]}...")
        else:
            print(f"  {component_value}")

    print("\n" + "=" * 70)

    # Test the optimized agent on holdout set
    print("\nEvaluating optimized agent on holdout test set...")
    print("=" * 70)

    from pydantic_ai import Agent
    from src.gepa.signature_agent import SignatureAgent
    from src.gepa.lm import get_openai_model

    # Create and configure agent
    test_agent = Agent(
        model=get_openai_model(config.agent_model),
        instructions=config.agent_instructions,
        output_type=IrisClassification,
    )

    test_signature_agent = SignatureAgent(
        test_agent,
        input_type=IrisInput,
    )

    # Track results
    correct_predictions = 0
    total_predictions = 0
    results_table = []

    # Apply optimized configuration and test on holdout set
    with result.apply_best_to(agent=test_agent, input_type=IrisInput):
        for i, test_record in enumerate(holdout_data, 1):
            # Create input from test record
            test_input = IrisInput(
                sepal_length=test_record["sepal_length"],
                sepal_width=test_record["sepal_width"],
                petal_length=test_record["petal_length"],
                petal_width=test_record["petal_width"],
            )

            # Get ground truth
            actual = test_record["label"]

            # Run prediction
            try:
                test_result = test_signature_agent.run_signature_sync(test_input)
                predicted = test_result.output.species
                confidence = test_result.output.confidence
                reasoning = test_result.output.reasoning

                # Check if correct
                is_correct = predicted == actual
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1

                # Store result
                results_table.append(
                    {
                        "case": i,
                        "sepal_length": test_input.sepal_length,
                        "sepal_width": test_input.sepal_width,
                        "petal_length": test_input.petal_length,
                        "petal_width": test_input.petal_width,
                        "predicted": predicted,
                        "actual": actual,
                        "confidence": confidence,
                        "correct": is_correct,
                        "reasoning": reasoning,
                    }
                )

            except Exception as e:
                print(f"\n⚠️  Error on test case {i}: {e}")
                results_table.append(
                    {
                        "case": i,
                        "sepal_length": test_input.sepal_length,
                        "sepal_width": test_input.sepal_width,
                        "petal_length": test_input.petal_length,
                        "petal_width": test_input.petal_width,
                        "predicted": "ERROR",
                        "actual": actual,
                        "confidence": 0.0,
                        "correct": False,
                        "reasoning": str(e),
                    }
                )

    # Print results table
    print(f"\nHoldout Test Results ({len(holdout_data)} flowers):")
    print("-" * 120)
    print(
        f"{'#':<4} {'SepL':<6} {'SepW':<6} {'PetL':<6} {'PetW':<6} {'Predicted':<12} {'Actual':<12} {'Conf':<6} {'Result':<8}"
    )
    print("-" * 120)

    for row in results_table:
        result_symbol = "✓" if row["correct"] else "✗"
        print(
            f"{row['case']:<4} {row['sepal_length']:<6.1f} {row['sepal_width']:<6.1f} "
            f"{row['petal_length']:<6.1f} {row['petal_width']:<6.1f} "
            f"{row['predicted']:<12} {row['actual']:<12} {row['confidence']:<6.2f} {result_symbol:<8}"
        )

    print("-" * 120)

    # Calculate and display accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(
            f"\n📊 Holdout Test Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})"
        )

    # Calculate per-species accuracy
    species_stats = {}
    for row in results_table:
        species = row["actual"]
        if species not in species_stats:
            species_stats[species] = {"correct": 0, "total": 0}
        species_stats[species]["total"] += 1
        if row["correct"]:
            species_stats[species]["correct"] += 1

    print("\nPer-Species Accuracy:")
    for species, stats in sorted(species_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"  {species.capitalize():<12}: {acc:.2%} ({stats['correct']}/{stats['total']})"
        )

    # Show a few detailed examples
    print("\n" + "=" * 70)
    print("Sample Detailed Predictions:")
    print("=" * 70)

    for i, row in enumerate(results_table[:5], 1):  # Show first 5
        result_symbol = "✓ CORRECT" if row["correct"] else "✗ INCORRECT"
        print(f"\n--- Case {row['case']} ({result_symbol}) ---")
        print(
            f"Measurements: Sepal L={row['sepal_length']:.1f}cm W={row['sepal_width']:.1f}cm, "
            f"Petal L={row['petal_length']:.1f}cm W={row['petal_width']:.1f}cm"
        )
        print(
            f"Predicted: {row['predicted'].upper()} (confidence: {row['confidence']:.2%})"
        )
        print(f"Actual: {row['actual'].upper()}")
        print(
            f"Reasoning: {row['reasoning'][:150]}..."
            if len(row["reasoning"]) > 150
            else f"Reasoning: {row['reasoning']}"
        )

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70 + "\n")

    return result


if __name__ == "__main__":
    # Run the example
    result = main()
