import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

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


def train_and_eval(config: CzechFinancialConfig, tables):
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
            ("cat", TargetEncoder(target_type="binary", smooth=10.0, random_state=config.random_state), cat_cols),
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

    tree = DecisionTreeClassifier(random_state=config.random_state)

    zero_r_baseline = DummyClassifier(strategy="most_frequent")

    # 7. Build Pipelines
    models = {
        "Baseline (Majority Class)": zero_r_baseline,
        "L1 Logistic Regression": Pipeline([("preprocess", preprocessor), ("classifier", lasso_lr)]),
        "L1 Linear SVC": Pipeline([("preprocess", preprocessor), ("classifier", linear_svc)]),
        "Decistion Tree": Pipeline([("preprocess", preprocessor), ("classifier", tree)]),
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
        "Decistion Tree": {"classifier__max_depth": list(range(3, 15))},
    }

    name_mappings = {"L1 Logistic Regression": "logistic", "L1 Linear SVC": "svc"}

    target_name_mappings = {"finished_binary": "finished", "all_binary": "all"}

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
            estimator=pipeline, param_grid=param_grids[name], cv=cv, scoring="average_precision", n_jobs=-1
        )

        # Fit the grid search strictly on the TRAINING set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            grid_search.fit(X_train, y_train)

        # Store the optimized pipeline
        best_models[name] = grid_search.best_estimator_

        print(f"Best CV PR-AUC: {grid_search.best_score_:.4f}")
        print(f"Best Parameters: {grid_search.best_params_}")

    print("\n--- Baseline CV Evaluation ---")
    baseline_scores = cross_val_score(
        models["Baseline (Majority Class)"], X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1
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
        test_score = average_precision_score(y_test, y_pred)
        print(f"{name}: PR-AUC = {test_score:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    # 12. Feature Importance Analysis
    # ---------------------------------------------------------
    print("\n=======================================================")
    print("                 Feature Importance Analysis           ")
    print("=======================================================")

    with open(f"data_{config.target_mode}.py", "w") as f:
        for name, pipeline in best_models.items():
            if name == "Baseline (Majority Class)":
                continue

            if name == "Decistion Tree":
                print(f"\n--- {name} Interactive Plotly Export ---")

                # 1. Extract model components
                preprocessor = pipeline.named_steps["preprocess"]
                feature_names = list(preprocessor.get_feature_names_out())
                tree_model = pipeline.named_steps["classifier"]

                # Extract tree structures
                children_left = tree_model.tree_.children_left
                children_right = tree_model.tree_.children_right
                features = tree_model.tree_.feature
                thresholds = tree_model.tree_.threshold.copy()  # copy to avoid mutating the original model
                values = tree_model.tree_.value

                # 2. Unscale continuous thresholds for tooltips
                scaler = preprocessor.named_transformers_["cont"]
                cont_cols = [cols for n, trans, cols in preprocessor.transformers_ if n == "cont"][0]

                for i, feat_idx in enumerate(features):
                    if feat_idx >= 0:
                        feat_name = feature_names[feat_idx]
                        if feat_name.startswith("cont__"):
                            orig_col = feat_name.replace("cont__", "")
                            if orig_col in cont_cols:
                                scaler_idx = cont_cols.index(orig_col)
                                mean = scaler.mean_[scaler_idx]
                                scale = scaler.scale_[scaler_idx]
                                thresholds[i] = thresholds[i] * scale + mean

                # 3. Compute Node Coordinates (X, Y) dynamically
                coords = {}

                def compute_layout(node_id, depth=0, left_bound=0.0, right_bound=1.0):
                    if node_id == -1:
                        return

                    # X coordinate is the midpoint of the current boundary allocation
                    x = (left_bound + right_bound) / 2.0
                    y = -depth  # Root is at 0, deeper nodes go downwards
                    coords[node_id] = (x, y)

                    # Allocate space split for children
                    compute_layout(children_left[node_id], depth + 1, left_bound, x)
                    compute_layout(children_right[node_id], depth + 1, x, right_bound)

                compute_layout(0)  # Start from root node

                # 4. Separate nodes and build links (edges)
                edge_x, edge_y = [], []
                node_x, node_y = [], []
                hover_text, node_color = [], []

                for node_id in range(tree_model.tree_.node_count):
                    x, y = coords[node_id]
                    node_x.append(x)
                    node_y.append(y)

                    # Determine Node details
                    num_samples = tree_model.tree_.n_node_samples[node_id]
                    node_val = values[node_id][0]
                    # Dominant class index
                    class_idx = np.argmax(node_val)
                    node_color.append(class_idx)

                    if features[node_id] != -2:  # Internal split node
                        feat_name = feature_names[features[node_id]]
                        thresh = thresholds[node_id]
                        text = f"<b>Node {node_id} (Split)</b><br>Split: {feat_name} ≤ {thresh:.4f}<br>Samples: {num_samples}<br>Distribution: {node_val.tolist()}"  # noqa: E501 line too long
                    else:  # Leaf node
                        text = f"<b>Node {node_id} (Leaf)</b><br>Samples: {num_samples}<br>Final Distribution: {node_val.tolist()}<br>Predicted Class: {class_idx}"  # noqa: E501 line too long
                    hover_text.append(text)

                    # Add line configurations to children
                    left = children_left[node_id]
                    right = children_right[node_id]

                    for child in [left, right]:
                        if child != -1:
                            cx, cy = coords[child]
                            edge_x.extend([x, cx, None])  # None prevents drawing continuous artifacts
                            edge_y.extend([y, cy, None])

                # 5. Construct Plotly Figure
                fig = go.Figure()

                # Edges (Lines connecting decisions)
                fig.add_trace(
                    go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#999999", width=1.5), hoverinfo="none")
                )

                # Nodes (Scatter markers with interactive hover cards)
                fig.add_trace(
                    go.Scatter(
                        x=node_x,
                        y=node_y,
                        mode="markers",
                        marker=dict(
                            size=18,
                            color=node_color,
                            colorscale="Bluered",  # Blue for Class 0, Red for Class 1
                            line=dict(color="black", width=1),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )

                # Clean layout geometry for presentations
                fig.update_layout(
                    title=f"Interactive Decision - {config.target_mode}",
                    title_font_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                )

                # Save interactive artifact as HTML for Quarto integration
                fig.write_html(f"interactive_tree_{config.target_mode}.html", full_html=False, include_plotlyjs="cdn")
                print("Successfully compiled and saved 'interactive_tree.html'")
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
            # print(
            #     non_zero_df[["Feature", "Coefficient"]]
            #     .sort_values("Coefficient", ascending=False)
            #     .to_string(index=False, float_format=lambda x: f"{x:.25f}")
            # )

            sorted_non_zero_df = non_zero_df[["Feature", "Coefficient"]].sort_values("Coefficient", ascending=False)

            identifier = f"{target_name_mappings[config.target_mode]}_{name_mappings[name]}"

            feature_variable = f"features_{identifier} = ["
            coefs_variable = f"coefs_{identifier} = ["

            for idx, row in sorted_non_zero_df.iterrows():
                print(row["Feature"], row["Coefficient"])
                feature_variable += f"\n    '{row['Feature']}',"
                coefs_variable += f"\n    {row['Coefficient']},"

            # remove trailing comma
            feature_variable = feature_variable[:-1]
            coefs_variable = coefs_variable[:-1]

            feature_variable += "\n]\n"
            coefs_variable += "\n]\n"

            f.write(feature_variable)
            f.write(coefs_variable)
            f.write(f"data_{identifier} = (features_{identifier}, coefs_{identifier})\n")


def main():  #
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
