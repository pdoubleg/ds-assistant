from typing import Union

import optuna
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.tools import ToolDefinition
from rich.console import Console
from rich.panel import Panel
from sklearn.model_selection import train_test_split

from src.tune.prompts import SYSTEM_PROMPT

from .schema import AnalysisAndRecommendations, AutoMLDependencies, PythonCode
from .tools import (
    generate_search_space_from_code,
    hpo_profile_from_dataframe,
    render_estimator_params,
)
from .utils import (
    Markdown,
    get_model_pipeline,
    get_scorer_smart,
)

console = Console()

MAX_MESSAGE_HISTORY = 20


async def message_at_index_contains_tool_return_parts(
    messages: list[ModelMessage], index: int
) -> bool:
    return any(isinstance(part, ToolReturnPart) for part in messages[index].parts)


async def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    number_of_messages = len(messages)
    number_of_messages_to_keep = MAX_MESSAGE_HISTORY
    if number_of_messages <= number_of_messages_to_keep:
        return messages
    if await message_at_index_contains_tool_return_parts(
        messages, number_of_messages - number_of_messages_to_keep
    ):
        return messages
    return messages[-number_of_messages_to_keep:]


async def only_use_df_if_allowed(
    ctx: RunContext[AutoMLDependencies],
    tool_def: ToolDefinition,
) -> Union[ToolDefinition, None]:
    """Conditionally allow dataset analysis tool."""
    if ctx.deps.use_dataset_analysis:
        return tool_def
    else:
        return None


def get_analysis_and_recommendations_agent(model: str = "gpt-4.1-mini"):
    analysis_and_recommendations_agent = Agent(
        model=OpenAIChatModel(model),
        deps_type=AutoMLDependencies,
        output_type=AnalysisAndRecommendations,
        instructions=SYSTEM_PROMPT,
    )

    @analysis_and_recommendations_agent.tool(prepare=only_use_df_if_allowed)
    async def get_dataset_characteristics(ctx: RunContext[AutoMLDependencies]) -> str:
        """Get a summary of the user's dataset."""
        if ctx.deps.verbose >= 1:
            console.print("🔍 Calling dataset analysis tool...", style="blue")
        X = ctx.deps.dataset.drop(columns=[ctx.deps.target_column])
        y = ctx.deps.dataset[ctx.deps.target_column]
        profile = hpo_profile_from_dataframe(X, y, task=None, mode="thorough")
        output_string = profile.render_markdown_facts()

        # Display the dataset characteristics for verbosity level 3
        if ctx.deps.verbose >= 3:
            md = Markdown(output_string)
            console.print(
                Panel(
                    md,
                    title="Tool Call Result",
                    border_style="cyan",
                )
            )

        return output_string

    return analysis_and_recommendations_agent


def get_initial_search_space_agent(model: str = "gpt-4.1-mini"):
    initial_search_space_agent = Agent(
        model=OpenAIChatModel(model),
        deps_type=AutoMLDependencies,
        output_type=PythonCode,
        retries=5,
        instructions=SYSTEM_PROMPT,
    )

    @initial_search_space_agent.output_validator
    def validate_initial_space(
        ctx: RunContext[AutoMLDependencies], python_code: PythonCode
    ) -> PythonCode:
        """Validate that the initial search space contains the required function."""
        if "def define_search_space" not in python_code.code:
            raise ModelRetry("Please define `def define_search_space(trial):`")
        return python_code

    @initial_search_space_agent.output_validator
    def check_if_optuna_works(
        ctx: RunContext[AutoMLDependencies], python_code: PythonCode
    ) -> PythonCode:
        try:
            define_fn = generate_search_space_from_code(python_code.code)
        except Exception as e:
            raise ModelRetry(f"Error generating search space: {e}") from e

        try:
            X = ctx.deps.dataset.drop(columns=[ctx.deps.target_column])
            y = ctx.deps.dataset[ctx.deps.target_column]

            # Take a small subset of data for quick validation
            if len(X) > 1000:
                X = X.iloc[:1000]
                y = y.iloc[:1000]

            scorer = get_scorer_smart(ctx.deps.metric, ctx.deps.task_type)

            def objective(trial: optuna.trial.Trial) -> float:
                model_hyperparameters = define_fn(trial)
                model = get_model_pipeline(
                    model_hyperparameters,
                    model_type=ctx.deps.estimator_type,
                    task_type=ctx.deps.task_type,
                    custom_estimator=ctx.deps.custom_estimator,
                )

                # Use simple train-test split instead of cross-validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                score = scorer(model, X_val, y_val)
                return float(score)

            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
            study = optuna.create_study(
                direction=ctx.deps.direction,
                sampler=optuna.samplers.RandomSampler(seed=42),
                study_name="validation_study",
                storage=None,
                load_if_exists=True,
            )
            study.optimize(objective, n_trials=1, gc_after_trial=True)
        except Exception as e:
            model = get_model_pipeline(
                model_type=ctx.deps.estimator_type,
                task_type=ctx.deps.task_type,
                custom_estimator=ctx.deps.custom_estimator,
            )
            valid_params = render_estimator_params(model)
            raise ModelRetry(
                f"Python code error: {e}\n\nPlease correct the code and try again.\nValid Parameters:\n{valid_params}"
            ) from e
        return python_code

    return initial_search_space_agent


def get_refine_search_space_agent(model: str = "gpt-4.1-mini"):
    refine_search_space_agent = Agent(
        model=OpenAIChatModel(model),
        deps_type=AutoMLDependencies,
        output_type=PythonCode,
        history_processors=[keep_recent_messages],
        instructions=SYSTEM_PROMPT,
        retries=5,
    )

    @refine_search_space_agent.output_validator
    def validate_refine_space(
        ctx: RunContext[AutoMLDependencies], python_code: PythonCode
    ) -> PythonCode:
        """Validate that the refine search space contains the required function."""
        if "def define_search_space" not in python_code.code:
            raise ModelRetry("Please define `def define_search_space(trial):`")
        return python_code

    @refine_search_space_agent.output_validator
    def check_if_optuna_works(
        ctx: RunContext[AutoMLDependencies], python_code: PythonCode
    ) -> PythonCode:
        try:
            define_fn = generate_search_space_from_code(python_code.code)
        except Exception as e:
            raise ModelRetry(f"Error generating search space: {e}") from e

        try:
            X = ctx.deps.dataset.drop(columns=[ctx.deps.target_column])
            y = ctx.deps.dataset[ctx.deps.target_column]

            # Take a small subset of data for quick validation
            if len(X) > 1000:
                X = X.iloc[:1000]
                y = y.iloc[:1000]

            scorer = get_scorer_smart(ctx.deps.metric, ctx.deps.task_type)

            def objective(trial: optuna.trial.Trial) -> float:
                model_hyperparameters = define_fn(trial)
                model = get_model_pipeline(
                    model_hyperparameters,
                    model_type=ctx.deps.estimator_type,
                    task_type=ctx.deps.task_type,
                    custom_estimator=ctx.deps.custom_estimator,
                )

                # Use simple train-test split instead of cross-validation
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                model.fit(X_train, y_train)
                score = scorer(model, X_val, y_val)
                return float(score)

            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
            study = optuna.create_study(
                direction=ctx.deps.direction,
                sampler=optuna.samplers.RandomSampler(seed=42),
                study_name="validation_study",
                storage=None,
                load_if_exists=True,
            )
            study.optimize(objective, n_trials=1, gc_after_trial=True)
        except Exception as e:
            model = get_model_pipeline(
                model_type=ctx.deps.estimator_type,
                task_type=ctx.deps.task_type,
                custom_estimator=ctx.deps.custom_estimator,
            )
            valid_params = render_estimator_params(model)
            raise ModelRetry(
                f"Python code error: {e}\n\nPlease correct the code and try again.\nValid Parameters:\n{valid_params}"
            ) from e
        return python_code

    return refine_search_space_agent
