from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_table.csv"


def load_data():
    print("Loading feature table...")
    df = pd.read_csv(FEATURE_PATH)
    print(f"Loaded {len(df):,} rows")
    print(f"Loaded {df.shape[1]} columns")

    return df


def split_features_target(df):
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]
    print(f"Features : {X.shape[1]}")
    print(f"Samples  : {X.shape[0]}")

    return X, y


def identify_columns(X):
    categorical_cols = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()
    print(f"Numerical columns   : {len(numerical_cols)}")
    print(f"Categorical columns : {len(categorical_cols)}")

    return numerical_cols, categorical_cols


def build_preprocessor(numerical_cols, categorical_cols):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print("\nClass Distribution:")
    print(f"Train:")
    print(y_train.value_counts(normalize=True))
    print(f"\nTest:")
    print(y_test.value_counts(normalize=True))
    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


def save_artifacts(preprocessor, X_train_processed, X_test_processed, X_train, X_test, y_train, y_test):
    MODELS_DIR = PROJECT_ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    print("Saved preprocessing pipeline.")
    feature_names = preprocessor.get_feature_names_out()
    print("Saving feature names...")
    joblib.dump(
        feature_names,
        MODELS_DIR / "feature_names.joblib"
    )
    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )
    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train_processed.to_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test_processed.to_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train.to_frame(name="TARGET").to_parquet(
        PROCESSED_DIR / "y_train.parquet"
    )
    y_test.to_frame(name="TARGET").to_parquet(
        PROCESSED_DIR / "y_test.parquet"
    )
    print("\nSaved processed datasets.")


def main():
    df = load_data()
    X, y = split_features_target(df)
    numerical_cols, categorical_cols = identify_columns(X)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    print("Fitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print("Preprocessing complete.")
    save_artifacts(
        preprocessor,
        X_train_processed,
        X_test_processed,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    print("\nSummary")
    print(f"Train shape : {X_train_processed.shape}")
    print(f"Test shape  : {X_test_processed.shape}")
    print("Preprocessing pipeline finished successfully.")


if __name__ == "__main__":
    main()
