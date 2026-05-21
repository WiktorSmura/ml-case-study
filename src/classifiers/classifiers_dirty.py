import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.svm import LinearSVC

# add src dir to imports
# Find project root by walking upward until pyproject.toml is found
PROJECT_ROOT = Path.cwd().resolve()

while PROJECT_ROOT != PROJECT_ROOT.parent and not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.czech_financial_preprocessing_pipeline import (  # noqa: E402 - import not at the top, for now a dirty fix with this path insert above
    CzechFinancialConfig,
    build_classification_dataset,
    load_and_normalize_tables,
    temporal_train_test_split,
)

DATA_DIR = Path("data/raw")
assert DATA_DIR.exists()


def grid_values():
    return [
        *[0.001 * (x + 1) for x in range(9)],
        *[0.01 * (x + 1) for x in range(9)],
        *[0.1 * (x + 1) for x in range(9)],
        *[1.0 * (x + 1) for x in range(9)],
        *[10.0 * (x + 1) for x in range(9)],
        *[10.0 * (x + 1) + 5.0 for x in range(9)],
        *[100.0 * (x + 1) for x in range(9)],
        *[100.0 * (x + 1) + 50.0 for x in range(9)],
    ]


def train_and_eval(config, tables):
    X, y, meta = build_classification_dataset(config, tables=tables)

    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    # 1. Clean the leaked index column if it exists
    if "Unnamed: 0" in X.columns:
        X = X.drop(columns=["Unnamed: 0"])

    # 2. Dynamically route the columns based on type
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    is_binary = [len(X[col].dropna().unique()) <= 2 for col in num_cols]
    cont_cols = [col for col, binary in zip(num_cols, is_binary) if not binary]
    bin_cols = [col for col, binary in zip(num_cols, is_binary) if binary]

    print(f"Categorical (Strings): {len(cat_cols)}")
    print(f"Continuous: {len(cont_cols)}")
    print(f"Binary (OHE/Flags): {len(bin_cols)}")

    # 3. Temporal Train/Test Split
    X_train, X_test, y_train, y_test, meta_train, meta_test = temporal_train_test_split(X, y, meta, test_size=0.20)

    print("\n--- Split Shapes ---")
    print(f"Train X: {X_train.shape}, Train y: {y_train.shape}")
    print(f"Test X: {X_test.shape}, Test y: {y_test.shape}")

    # 4. Build the Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", TargetEncoder(target_type="binary", smooth=10.0), cat_cols),
            ("cont", StandardScaler(), cont_cols),
            ("bin", "passthrough", bin_cols),
        ]
    )

    # 6. Define Classifiers
    # Using elasticnet with l1_ratio=1.0 safely maps to pure L1 without warnings in saga
    lasso_lr = LogisticRegression(
        l1_ratio=1.0, solver="saga", C=1.0, max_iter=15000, class_weight="balanced", random_state=config.random_state
    )

    linear_svc = LinearSVC(
        penalty="l1", dual=False, C=0.1, class_weight="balanced", max_iter=15000, random_state=config.random_state
    )

    ridge_clf = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=config.random_state)

    zero_r_baseline = DummyClassifier(strategy="most_frequent")

    # 7. Build Pipelines
    models = {
        "Baseline (Majority Class)": zero_r_baseline,
        "L1 Logistic Regression": Pipeline([("preprocess", preprocessor), ("classifier", lasso_lr)]),
        "L1 Linear SVC": Pipeline([("preprocess", preprocessor), ("classifier", linear_svc)]),
        "L2 Ridge Classifier": Pipeline([("preprocess", preprocessor), ("classifier", ridge_clf)]),
    }

    # 8. Evaluate Setup (CV strictly on Training Data)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    # ---------------------------------------------------------
    # 9. Hyperparameter Tuning Grids
    # ---------------------------------------------------------
    param_grids = {
        "L1 Logistic Regression": {
            "classifier__C": grid_values(),
        },
        "L1 Linear SVC": {
            "classifier__C": grid_values(),
        },
        "L2 Ridge Classifier": {
            "classifier__alpha": grid_values(),
        },
    }

    # ---------------------------------------------------------
    # 10. Execute Grid Search on Train Data
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("                 Hyperparameter Tuning (Train)         ")
    print("=======================================================")

    best_models = {}

    for name, pipeline in models.items():
        if name == "Baseline (Majority Class)":
            continue

        print(f"\n--- Tuning {name} ---")
        grid_search = GridSearchCV(
            estimator=pipeline, param_grid=param_grids[name], cv=cv, scoring="balanced_accuracy", n_jobs=-1
        )

        # Fit the grid search strictly on the TRAINING set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            grid_search.fit(X_train, y_train)

        # Store the optimized pipeline
        best_models[name] = grid_search.best_estimator_

        print(f"Best CV Balanced Accuracy: {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

    print("\n--- Baseline CV Evaluation ---")
    baseline_scores = cross_val_score(
        models["Baseline (Majority Class)"], X_train, y_train, cv=cv, scoring="balanced_accuracy", n_jobs=-1
    )
    # Fit baseline to use it in final testing
    best_models["Baseline (Majority Class)"] = models["Baseline (Majority Class)"].fit(X_train, y_train)
    print(f"Baseline (Majority Class): Mean={baseline_scores.mean():.4f}, Std={baseline_scores.std():.4f}")

    # ---------------------------------------------------------
    # 11. Final Evaluation on Held-Out Test Data
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("                 Final Test Set Evaluation             ")
    print("=======================================================")

    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        test_score = balanced_accuracy_score(y_test, y_pred)
        print(f"{name}: Balanced Accuracy = {test_score:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    # 12. Feature Importance Analysis
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("                 Feature Importance Analysis           ")
    print("=======================================================")

    for name, pipeline in best_models.items():
        if name == "Baseline (Majority Class)":
            continue

        print(f"\n--- {name} Top 10 Features ---")

        # 1. Get feature names out of the ColumnTransformer
        preprocessor = pipeline.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()

        # 3. Extract the coefficients from the classifier
        classifier = pipeline.named_steps["classifier"]
        coefficients = classifier.coef_.flatten()

        # 4. Build a DataFrame for easy sorting
        feature_importance = pd.DataFrame(
            {"Feature": feature_names, "Coefficient": coefficients, "Abs_Coefficient": np.abs(coefficients)}
        )

        # Sort by the absolute magnitude of the coefficient
        feature_importance = feature_importance.sort_values(by="Abs_Coefficient", ascending=False)

        # Count how many features the model actually used
        non_zero_count = (feature_importance["Coefficient"] != 0).sum()
        print(f"Features utilized (non-zero): {non_zero_count} out of {len(feature_names)}")

        non_zero_df = feature_importance[feature_importance["Coefficient"] != 0]
        # Display the top 10 most impactful features
        print(
            non_zero_df[["Feature", "Coefficient"]]
            .sort_values("Coefficient", ascending=False)
            .to_string(index=False, float_format=lambda x: f"{x:.25f}")
        )


def main():
    tables = load_and_normalize_tables(DATA_DIR)

    config = CzechFinancialConfig(
        data_dir=DATA_DIR,
        target_mode="finished_binary",
        transaction_windows_days=(30, 90, 180, 365),
        include_full_history=True,
        include_orders=False,  # run an ablation later with False
        include_cards=True,
        include_district=True,
        include_owner_client=False,  # client = owner in this dataset
        strict_pre_loan_transactions=True,  # trans_date < loan_date
        min_category_count=5,
        random_state=42,
        drop_duplicate_features=True,
        drop_constant_features=True,
        drop_static_redundant_features=True,
        strict_pre_loan_cards=True,
        missing_value_policy="explicit",
    )

    print("---------------------------------------------")
    print("       USING JUST FINISHED TRANSACTIONS      ")
    print("---------------------------------------------")
    train_and_eval(config, tables)

    config = CzechFinancialConfig(
        data_dir=DATA_DIR,
        target_mode="all_binary",
        transaction_windows_days=(30, 90, 180, 365),
        include_full_history=True,
        include_orders=False,  # run an ablation later with False
        include_cards=True,
        include_district=True,
        include_owner_client=False,  # client = owner in this dataset
        strict_pre_loan_transactions=True,  # trans_date < loan_date
        min_category_count=5,
        random_state=42,
        drop_duplicate_features=True,
        drop_constant_features=True,
        drop_static_redundant_features=True,
        strict_pre_loan_cards=True,
        missing_value_policy="explicit",
    )

    print("---------------------------------------------")
    print("            USING ALL TRANSACTIONS           ")
    print("---------------------------------------------")
    train_and_eval(config, tables)


if __name__ == "__main__":
    main()
