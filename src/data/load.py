from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_TRANSACTION_COLUMNS = ("TransactionID", "TransactionDT", "isFraud")


class DataValidationError(ValueError):
    """Raised when raw fraud data violates required structural rules."""


def read_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Read the project YAML configuration."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_transaction_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the labeled transaction table."""
    transaction_df = pd.read_csv(path)
    validate_transaction_data(transaction_df)
    return transaction_df


def load_identity_data(path: str | Path | None) -> pd.DataFrame | None:
    """Load and validate the optional identity table."""
    if path is None:
        return None
    identity_path = Path(path)
    if not identity_path.exists():
        return None
    identity_df = pd.read_csv(identity_path)
    validate_identity_data(identity_df)
    return identity_df


def validate_transaction_data(df: pd.DataFrame) -> None:
    """Validate required transaction data invariants before splitting."""
    if df.empty:
        raise DataValidationError("Transaction data is empty.")

    missing_columns = [col for col in REQUIRED_TRANSACTION_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Transaction data is missing required columns: {missing_columns}."
        )

    _validate_transaction_id(df, "transaction data")

    if df["TransactionDT"].isna().any():
        raise DataValidationError("TransactionDT contains null values.")

    transaction_dt = pd.to_numeric(df["TransactionDT"], errors="coerce")
    if transaction_dt.isna().any():
        raise DataValidationError("TransactionDT contains invalid non-numeric values.")

    unique_targets = set(df["isFraud"].dropna().unique())
    if df["isFraud"].isna().any() or not unique_targets.issubset({0, 1}):
        raise DataValidationError("isFraud must contain only 0 and 1 values.")


def validate_identity_data(df: pd.DataFrame) -> None:
    """Validate required identity data invariants before joining."""
    if df.empty:
        raise DataValidationError("Identity data is empty.")

    if "TransactionID" not in df.columns:
        raise DataValidationError("Identity data is missing required column: TransactionID.")

    _validate_transaction_id(df, "identity data")


def merge_transaction_identity(
    transaction_df: pd.DataFrame, identity_df: pd.DataFrame | None
) -> pd.DataFrame:
    """Left join identity fields to transactions while preserving transaction rows."""
    validate_transaction_data(transaction_df)

    if identity_df is None:
        merged_df = transaction_df.copy()
    else:
        validate_identity_data(identity_df)
        transaction_rows = len(transaction_df)
        merged_df = transaction_df.merge(
            identity_df,
            how="left",
            on="TransactionID",
            suffixes=("", "_identity"),
            validate="one_to_one",
        )
        if len(merged_df) != transaction_rows:
            raise DataValidationError(
                "Merged row count does not equal transaction row count: "
                f"{len(merged_df)} != {transaction_rows}."
            )

    if not merged_df["TransactionID"].is_unique:
        raise DataValidationError("TransactionID is not unique after merge.")

    return merged_df


def load_labeled_data(config_path: str | Path = "configs/config.yaml") -> tuple[pd.DataFrame, int, int]:
    """Load, validate, and merge the labeled IEEE-CIS transaction and identity files."""
    config = read_config(config_path)
    transaction_df = load_transaction_data(config["data"]["train_transaction"])
    identity_df = load_identity_data(config["data"].get("train_identity"))
    merged_df = merge_transaction_identity(transaction_df, identity_df)
    return merged_df, len(transaction_df), 0 if identity_df is None else len(identity_df)


def _validate_transaction_id(df: pd.DataFrame, table_name: str) -> None:
    if df["TransactionID"].isna().any():
        raise DataValidationError(f"TransactionID contains null values in {table_name}.")
    if not df["TransactionID"].is_unique:
        raise DataValidationError(f"TransactionID contains duplicate values in {table_name}.")
