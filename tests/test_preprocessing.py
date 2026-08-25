from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import FraudDataPreprocessor, load_preprocessor


def make_train_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5],
            "TransactionDT": [10, 20, 30, 40, 50],
            "isFraud": [0, 0, 1, 0, 1],
            "amount": [10.0, np.nan, 30.0, 40.0, 50.0],
            "all_missing_in_train": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "mostly_present_in_train": [1.0, 2.0, np.nan, 4.0, 5.0],
            "category": ["a", "b", None, "a", "c"],
        }
    )


def make_validation_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [6, 7],
            "TransactionDT": [60, 70],
            "isFraud": [0, 1],
            "amount": [np.nan, 1000.0],
            "all_missing_in_train": [1.0, 2.0],
            "mostly_present_in_train": [100.0, 200.0],
            "category": ["unseen", None],
        }
    )


def make_test_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [8, 9],
            "TransactionDT": [80, 90],
            "isFraud": [0, 1],
            "amount": [5.0, np.nan],
            "all_missing_in_train": [3.0, 4.0],
            "mostly_present_in_train": [300.0, 400.0],
            "category": ["another_unseen", "b"],
        }
    )


def test_high_missing_columns_are_determined_from_training_data_only() -> None:
    preprocessor = FraudDataPreprocessor(missing_threshold=0.95).fit(make_train_features())

    assert preprocessor.dropped_columns == ["all_missing_in_train"]
    assert "all_missing_in_train" not in preprocessor.feature_columns


def test_numerical_medians_are_learned_from_training_data_only() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    assert preprocessor.numeric_medians["amount"] == 35.0
    transformed = preprocessor.transform(make_validation_features())
    amount_index = preprocessor.feature_columns.index("amount")
    assert transformed[0, amount_index] == 35.0


def test_numerical_missing_values_are_handled() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    transformed = preprocessor.transform(make_validation_features())

    assert np.isfinite(transformed).all()


def test_categorical_missing_values_are_handled() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    transformed = preprocessor.transform(make_validation_features())
    category_index = preprocessor.feature_columns.index("category")

    assert np.isfinite(transformed[:, category_index]).all()


def test_unseen_categories_in_validation_do_not_crash() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    transformed = preprocessor.transform(make_validation_features())
    category_index = preprocessor.feature_columns.index("category")

    assert transformed[0, category_index] == -1


def test_unseen_categories_in_test_do_not_crash() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    transformed = preprocessor.transform(make_test_features())
    category_index = preprocessor.feature_columns.index("category")

    assert transformed[0, category_index] == -1


def test_transformations_have_consistent_feature_dimensions() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    train = preprocessor.transform(make_train_features())
    validation = preprocessor.transform(make_validation_features())
    test = preprocessor.transform(make_test_features())

    assert train.shape[1] == validation.shape[1] == test.shape[1]


def test_feature_ordering_remains_consistent() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())
    shuffled_columns = make_validation_features()[
        ["category", "mostly_present_in_train", "all_missing_in_train", "amount", "isFraud", "TransactionDT", "TransactionID"]
    ]

    normal = preprocessor.transform(make_validation_features())
    shuffled = preprocessor.transform(shuffled_columns)

    np.testing.assert_array_equal(normal, shuffled)


def test_fitted_preprocessor_can_be_saved_and_loaded_using_joblib(tmp_path) -> None:
    path = tmp_path / "preprocessor.joblib"
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    preprocessor.save(path)
    loaded = load_preprocessor(path)

    assert isinstance(joblib.load(path), FraudDataPreprocessor)
    np.testing.assert_array_equal(
        preprocessor.transform(make_validation_features()),
        loaded.transform(make_validation_features()),
    )


def test_metadata_exposes_required_fields() -> None:
    preprocessor = FraudDataPreprocessor().fit(make_train_features())

    metadata = preprocessor.get_metadata()

    assert set(metadata) >= {
        "dropped_columns",
        "numeric_columns",
        "categorical_columns",
        "final_feature_count",
        "missing_threshold",
        "excluded_columns",
    }
