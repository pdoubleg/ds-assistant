import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def default_embedding_dim(cardinality: int) -> int:
    """Return a compact embedding size for a categorical feature.

    Args:
        cardinality: Number of integer category ids, including reserved ids.

    Returns:
        Embedding width to use for the categorical feature.

    Example:
        >>> default_embedding_dim(10)
        5
    """
    if cardinality < 1:
        raise ValueError("Categorical cardinality must be at least 1.")
    return min(50, max(1, (cardinality + 1) // 2))


class ResidualBlock(nn.Module):
    """Residual MLP block for tabular features."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual transformation."""
        return self.norm(x + self.block(x))


class RealMLP(nn.Module):
    """Residual MLP for numeric and categorical tabular features.

    Categorical inputs are integer encoded and passed through one embedding
    layer per feature before being concatenated with numeric inputs.

    Example:
        >>> model = RealMLP(input_dim=3, category_cardinalities=[4, 10])
        >>> x_num = torch.randn(8, 3)
        >>> x_cat = torch.randint(0, 4, (8, 2))
        >>> logits = model(x_num, x_cat)
    """

    def __init__(
        self,
        input_dim: int = 0,
        category_cardinalities: list[int] | tuple[int, ...] | None = None,
        embedding_dims: list[int] | tuple[int, ...] | None = None,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.numeric_input_dim = input_dim
        self.category_cardinalities = list(category_cardinalities or [])

        if input_dim < 0:
            raise ValueError("input_dim must be non-negative.")
        if any(cardinality < 1 for cardinality in self.category_cardinalities):
            raise ValueError("All category cardinalities must be at least 1.")

        if embedding_dims is None:
            self.embedding_dims = [
                default_embedding_dim(cardinality)
                for cardinality in self.category_cardinalities
            ]
        else:
            self.embedding_dims = list(embedding_dims)
            if len(self.embedding_dims) != len(self.category_cardinalities):
                raise ValueError(
                    "embedding_dims must have one value per category cardinality."
                )
            if any(dim < 1 for dim in self.embedding_dims):
                raise ValueError("All embedding dimensions must be at least 1.")

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dim)
                for cardinality, embedding_dim in zip(
                    self.category_cardinalities,
                    self.embedding_dims,
                    strict=True,
                )
            ]
        )
        self.embedding_output_dim = sum(self.embedding_dims)
        total_input_dim = input_dim + self.embedding_output_dim

        if total_input_dim < 1:
            raise ValueError(
                "RealMLP needs at least one numeric or categorical feature."
            )

        self.input_layer = nn.Linear(total_input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return raw logits, not probabilities."""
        x = self._combine_inputs(x_num, x_cat)
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)

    def _combine_inputs(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor | None,
    ) -> torch.Tensor:
        """Concatenate numeric features and categorical embeddings."""
        pieces: list[torch.Tensor] = []

        if x_num.ndim != 2:
            raise ValueError("x_num must be a 2D tensor.")
        if x_num.shape[1] != self.numeric_input_dim:
            raise ValueError(
                f"x_num has {x_num.shape[1]} columns, expected {self.numeric_input_dim}."
            )
        if self.numeric_input_dim:
            pieces.append(x_num)

        if self.embeddings:
            if x_cat is None:
                raise ValueError(
                    "x_cat is required when category_cardinalities are set."
                )
            if x_cat.ndim != 2:
                raise ValueError("x_cat must be a 2D tensor.")
            if x_cat.shape[1] != len(self.embeddings):
                raise ValueError(
                    f"x_cat has {x_cat.shape[1]} columns, expected {len(self.embeddings)}."
                )

            # Each categorical column owns its own embedding table.
            embedded_columns = [
                embedding(x_cat[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ]
            pieces.append(torch.cat(embedded_columns, dim=1))

        return torch.cat(pieces, dim=1)


def _feature_tensors(
    x: pd.DataFrame | dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convert model features into numeric and optional categorical tensors.

    Args:
        x: Either a numeric dataframe or the dictionary returned by
            ``TabularTorchPreprocessor.transform``.

    Returns:
        Numeric float tensor and optional categorical integer tensor.
    """
    if isinstance(x, pd.DataFrame):
        x_num = x.to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(x_num), None

    if not isinstance(x, dict):
        raise TypeError(
            "x must be a pandas DataFrame or a dict with x_num/x_cat arrays."
        )
    if "x_num" not in x:
        raise ValueError("Feature dict must contain an 'x_num' array.")

    x_num = np.asarray(x["x_num"], dtype=np.float32)
    x_cat = np.asarray(x.get("x_cat", np.empty((len(x_num), 0))), dtype=np.int64)

    if x_num.ndim != 2:
        raise ValueError("x_num must be a 2D array.")
    if x_cat.ndim != 2:
        raise ValueError("x_cat must be a 2D array.")
    if len(x_num) != len(x_cat):
        raise ValueError("x_num and x_cat must have the same number of rows.")

    cat_tensor = torch.from_numpy(x_cat) if x_cat.shape[1] else None
    return torch.from_numpy(x_num), cat_tensor


def make_binary_dataloader(
    x: pd.DataFrame | dict[str, np.ndarray],
    y: pd.Series | pd.DataFrame,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    """Convert pandas train data into a PyTorch DataLoader.

    Args:
        x: Numeric feature dataframe, or a feature dictionary with ``x_num`` and
            optional ``x_cat`` arrays from ``TabularTorchPreprocessor``.
        y: Binary target encoded as 0/1.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle rows each epoch.

    Returns:
        DataLoader for binary classification.
    """
    X_num, X_cat = _feature_tensors(x)

    # Accept either a Series or a single-column DataFrame.
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a Series or a single-column DataFrame.")
        y = y.iloc[:, 0]

    # BCEWithLogitsLoss expects float targets; reshape to (n, 1)
    # so target and model output have identical shapes.
    y_np = y.to_numpy(dtype=np.float32, copy=True).reshape(-1, 1)

    y_tensor = torch.from_numpy(y_np)

    if len(X_num) != len(y_tensor):
        raise ValueError("x and y must have the same number of rows.")

    if X_cat is None:
        dataset = TensorDataset(X_num, y_tensor)
    else:
        dataset = TensorDataset(X_num, X_cat, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_binary_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
) -> None:
    """Train a binary classifier with BCEWithLogitsLoss.

    Args:
        model: PyTorch model returning one logit per example.
        dataloader: Training batches.
        epochs: Number of epochs.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        device: Training device.
    """
    device = torch.device(device)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            if len(batch) == 2:
                X_batch, y_batch = batch
                X_cat_batch = None
            elif len(batch) == 3:
                X_batch, X_cat_batch, y_batch = batch
                X_cat_batch = X_cat_batch.to(device)
            else:
                raise ValueError("Expected dataloader batches with 2 or 3 tensors.")

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if X_cat_batch is None:
                logits = model(X_batch)  # shape: (batch_size, 1)
            else:
                logits = model(X_batch, X_cat_batch)  # shape: (batch_size, 1)
            loss = loss_fn(logits, y_batch)  # y_batch shape: (batch_size, 1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1:02d} | loss={avg_loss:.4f}")


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    x: pd.DataFrame | dict[str, np.ndarray],
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Return positive-class probabilities for binary classification."""
    device = torch.device(device)
    model.eval()
    model.to(device)

    X_num, X_cat = _feature_tensors(x)
    X_num = X_num.to(device)
    X_cat = X_cat.to(device) if X_cat is not None else None

    if X_cat is None:
        logits = model(X_num)
    else:
        logits = model(X_num, X_cat)
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
    return probs


__all__ = [
    "RealMLP",
    "default_embedding_dim",
    "make_binary_dataloader",
    "predict_proba",
    "train_binary_classifier",
]
