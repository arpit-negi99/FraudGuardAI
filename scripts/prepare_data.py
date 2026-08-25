from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_labeled_data, read_config
from src.data.preprocess import FraudDataPreprocessor
from src.data.split import chronological_split, separate_features_target


def main() -> None:
    config = read_config(ROOT / "configs" / "config.yaml")

    print("FraudGuard AI - Data Preparation")
    print()

    merged_df, transaction_rows, identity_rows = load_labeled_data(ROOT / "configs" / "config.yaml")
    print(f"Loaded transaction rows: {transaction_rows}")
    print(f"Loaded identity rows: {identity_rows}")
    print(f"Merged labeled rows: {len(merged_df)}")
    print()

    split_config = config["split"]
    train_df, validation_df, test_df = chronological_split(
        merged_df,
        train_ratio=split_config["train_ratio"],
        validation_ratio=split_config["validation_ratio"],
        test_ratio=split_config["test_ratio"],
    )

    target_column = config["target"]["column"]
    X_train, y_train, X_validation, y_validation, X_test, y_test = separate_features_target(
        train_df, validation_df, test_df, target_column=target_column
    )

    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Held-out test rows: {len(test_df)}")
    print()
    print(f"Train fraud rate: {y_train.mean():.6f}")
    print(f"Validation fraud rate: {y_validation.mean():.6f}")
    print(f"Test fraud rate: {y_test.mean():.6f}")
    print()

    preprocessor = FraudDataPreprocessor(
        missing_threshold=config["preprocessing"]["missing_column_threshold"]
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)
    X_test_transformed = preprocessor.transform(X_test)

    if not (
        X_train_transformed.shape[1]
        == X_validation_transformed.shape[1]
        == X_test_transformed.shape[1]
    ):
        raise RuntimeError("Transformed feature dimensions are inconsistent across splits.")

    preprocessor.save(
        ROOT / "artifacts" / "preprocessors" / "preprocessor.joblib",
        ROOT / "artifacts" / "preprocessors" / "preprocessing_metadata.json",
    )

    print(f"Numerical features: {len(preprocessor.numeric_columns)}")
    print(f"Categorical features: {len(preprocessor.categorical_columns)}")
    print(f"Dropped high-missing columns: {len(preprocessor.dropped_columns)}")
    print(f"Final transformed features: {preprocessor.final_feature_count}")
    print()
    print("Preprocessor saved successfully.")


if __name__ == "__main__":
    main()
