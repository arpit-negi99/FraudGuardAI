from __future__ import annotations

import pandas as pd


EXCLUDED_FEATURE_COLUMNS = ("isFraud", "TransactionID", "TransactionDT")


class SplitValidationError(ValueError):
    """Raised when chronological split invariants are violated."""


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by TransactionDT and return contiguous train/validation/test splits."""
    _validate_split_inputs(df, train_ratio, validation_ratio, test_ratio)

    sorted_df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    row_count = len(sorted_df)
    train_end = int(row_count * train_ratio)
    validation_end = int(row_count * (train_ratio + validation_ratio))

    train_df = sorted_df.iloc[:train_end].copy()
    validation_df = sorted_df.iloc[train_end:validation_end].copy()
    test_df = sorted_df.iloc[validation_end:].copy()

    _validate_split_output(train_df, validation_df, test_df)
    return train_df, validation_df, test_df


def separate_features_target(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "isFraud",
    excluded_columns: tuple[str, ...] = EXCLUDED_FEATURE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create leakage-safe feature and target objects for each split."""
    for split_name, split_df in (
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        if target_column not in split_df.columns:
            raise SplitValidationError(f"{split_name} split is missing target column {target_column}.")

    feature_drop_columns = [col for col in excluded_columns if col in train_df.columns]
    X_train = train_df.drop(columns=feature_drop_columns)
    X_validation = validation_df.drop(columns=[col for col in excluded_columns if col in validation_df.columns])
    X_test = test_df.drop(columns=[col for col in excluded_columns if col in test_df.columns])

    return (
        X_train,
        train_df[target_column].copy(),
        X_validation,
        validation_df[target_column].copy(),
        X_test,
        test_df[target_column].copy(),
    )


def _validate_split_inputs(
    df: pd.DataFrame, train_ratio: float, validation_ratio: float, test_ratio: float
) -> None:
    if df.empty:
        raise SplitValidationError("Cannot split an empty DataFrame.")
    for column in ("TransactionID", "TransactionDT"):
        if column not in df.columns:
            raise SplitValidationError(f"Cannot split data without {column}.")
    if df["TransactionID"].duplicated().any():
        raise SplitValidationError("TransactionID values must be unique before splitting.")
    if df["TransactionDT"].isna().any():
        raise SplitValidationError("TransactionDT contains null values.")
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise SplitValidationError(f"Split ratios must sum to 1.0; got {ratio_sum}.")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise SplitValidationError("All split ratios must be positive.")


def _validate_split_output(
    train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    if train_df.empty or validation_df.empty or test_df.empty:
        raise SplitValidationError("Chronological split produced an empty partition.")

    if train_df["TransactionDT"].max() > validation_df["TransactionDT"].min():
        raise SplitValidationError("Train split occurs after validation split.")
    if validation_df["TransactionDT"].max() > test_df["TransactionDT"].min():
        raise SplitValidationError("Validation split occurs after test split.")

    train_ids = set(train_df["TransactionID"])
    validation_ids = set(validation_df["TransactionID"])
    test_ids = set(test_df["TransactionID"])
    if train_ids & validation_ids:
        raise SplitValidationError("TransactionID overlap exists between train and validation.")
    if train_ids & test_ids:
        raise SplitValidationError("TransactionID overlap exists between train and test.")
    if validation_ids & test_ids:
        raise SplitValidationError("TransactionID overlap exists between validation and test.")
