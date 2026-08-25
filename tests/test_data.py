from __future__ import annotations

import pandas as pd
import pytest

from src.data.load import (
    DataValidationError,
    merge_transaction_identity,
    validate_identity_data,
    validate_transaction_data,
)
from src.data.split import chronological_split, separate_features_target


def make_transaction_df(row_count: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": range(1, row_count + 1),
            "TransactionDT": list(reversed(range(100, 100 + row_count))),
            "isFraud": [0, 1] * (row_count // 2),
            "TransactionAmt": [10.0 + i for i in range(row_count)],
            "ProductCD": ["W", "C"] * (row_count // 2),
        }
    )


def make_identity_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 3, 5],
            "DeviceType": ["desktop", "mobile", "desktop"],
        }
    )


def test_duplicate_transaction_id_is_rejected() -> None:
    df = make_transaction_df()
    df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]

    with pytest.raises(DataValidationError, match="duplicate"):
        validate_transaction_data(df)


def test_duplicate_identity_transaction_id_is_rejected() -> None:
    df = make_identity_df()
    df.loc[1, "TransactionID"] = df.loc[0, "TransactionID"]

    with pytest.raises(DataValidationError, match="duplicate"):
        validate_identity_data(df)


def test_null_transaction_id_is_rejected() -> None:
    df = make_transaction_df()
    df.loc[0, "TransactionID"] = None

    with pytest.raises(DataValidationError, match="null"):
        validate_transaction_data(df)


def test_missing_required_transaction_columns_raise_descriptive_error() -> None:
    df = make_transaction_df().drop(columns=["TransactionDT"])

    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_transaction_data(df)


def test_invalid_is_fraud_values_are_rejected() -> None:
    df = make_transaction_df()
    df.loc[0, "isFraud"] = 2

    with pytest.raises(DataValidationError, match="isFraud"):
        validate_transaction_data(df)


def test_empty_required_dataframes_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        validate_transaction_data(make_transaction_df().iloc[0:0])

    with pytest.raises(DataValidationError, match="empty"):
        validate_identity_data(make_identity_df().iloc[0:0])


def test_left_join_preserves_transaction_row_count() -> None:
    transaction_df = make_transaction_df()
    identity_df = make_identity_df()

    merged_df = merge_transaction_identity(transaction_df, identity_df)

    assert len(merged_df) == len(transaction_df)


def test_transactions_without_identity_records_remain_present() -> None:
    transaction_df = make_transaction_df()
    identity_df = make_identity_df()

    merged_df = merge_transaction_identity(transaction_df, identity_df)

    missing_identity_row = merged_df.loc[merged_df["TransactionID"] == 2].iloc[0]
    assert pd.isna(missing_identity_row["DeviceType"])


def test_transaction_id_stays_unique_after_merge() -> None:
    merged_df = merge_transaction_identity(make_transaction_df(), make_identity_df())

    assert merged_df["TransactionID"].is_unique


def test_chronological_ordering_is_correct() -> None:
    train_df, validation_df, test_df = chronological_split(make_transaction_df(20))

    assert train_df["TransactionDT"].max() <= validation_df["TransactionDT"].min()
    assert validation_df["TransactionDT"].max() <= test_df["TransactionDT"].min()


def test_no_transaction_id_overlap_exists_across_splits() -> None:
    train_df, validation_df, test_df = chronological_split(make_transaction_df(20))

    assert set(train_df["TransactionID"]).isdisjoint(validation_df["TransactionID"])
    assert set(train_df["TransactionID"]).isdisjoint(test_df["TransactionID"])
    assert set(validation_df["TransactionID"]).isdisjoint(test_df["TransactionID"])


def test_split_proportions_are_approximately_70_15_15() -> None:
    train_df, validation_df, test_df = chronological_split(make_transaction_df(100))

    assert len(train_df) == 70
    assert len(validation_df) == 15
    assert len(test_df) == 15


def test_no_shuffling_occurs() -> None:
    train_df, validation_df, test_df = chronological_split(make_transaction_df(20))
    sorted_df = make_transaction_df(20).sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)

    combined_ids = pd.concat(
        [train_df["TransactionID"], validation_df["TransactionID"], test_df["TransactionID"]],
        ignore_index=True,
    )
    assert combined_ids.tolist() == sorted_df["TransactionID"].tolist()


def test_leakage_columns_are_excluded_from_model_features() -> None:
    train_df, validation_df, test_df = chronological_split(make_transaction_df(20))

    X_train, _, X_validation, _, X_test, _ = separate_features_target(
        train_df, validation_df, test_df
    )

    for features in (X_train, X_validation, X_test):
        assert "TransactionID" not in features.columns
        assert "TransactionDT" not in features.columns
        assert "isFraud" not in features.columns
