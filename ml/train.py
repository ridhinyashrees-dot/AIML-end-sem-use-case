import json
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from backend.config import PROC, MODELS, FEATURES, BLOCK_DEG, UserError


# ---------------------------------------------------------
# MODEL BUILDERS
# ---------------------------------------------------------

def build(name, params=None):

    params = params or {}

    if name == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", None),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                max_features=params.get("max_features", "sqrt"),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )

    if name == "svm":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                C=params.get("C", 1.0),
                gamma=params.get("gamma", "scale"),
                kernel=params.get("kernel", "rbf"),
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
        )

    if name == "xgboost":

        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise UserError(
                "XGBoost not installed. Run: pip install xgboost"
            )

        return make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", 4),
                learning_rate=params.get("learning_rate", 0.05),
                min_child_weight=params.get("min_child_weight", 1),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("colsample_bytree", 0.9),
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        )

    raise UserError(f"Unknown model '{name}'.")


# ---------------------------------------------------------
# HYPERPARAMETER SEARCH SPACE
# ---------------------------------------------------------

PARAM_GRIDS = {

    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
    },

    "svm": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.1],
        "kernel": ["rbf"],
    },

    "xgboost": {
        "n_estimators": [200, 400],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [1, 3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
}


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def compute_metrics(y_true, probabilities, threshold=0.5):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": round(float(
            accuracy_score(y_true, predictions)
        ), 4),

        "precision": round(float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ), 4),

        "recall": round(float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ), 4),

        "f1": round(float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ), 4),

        "roc_auc": round(float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ), 4),

        "confusion": [
            [int(tn), int(fp)],
            [int(fn), int(tp)]
        ]
    }


# ---------------------------------------------------------
# OOF PREDICTIONS
# ---------------------------------------------------------

def get_oof_predictions(model_name, params, X, y, groups, k):

    oof = np.zeros(len(X))

    cv = GroupKFold(n_splits=k)

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups
    ):

        model = build(
            model_name,
            params
        )

        model.fit(
            X.iloc[train_idx],
            y[train_idx]
        )

        oof[test_idx] = model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]

    return oof


# ---------------------------------------------------------
# THRESHOLD OPTIMIZATION
# ---------------------------------------------------------

def find_best_threshold(y, probabilities):

    best_threshold = 0.5
    best_f1 = -1

    thresholds = np.arange(
        0.20,
        0.81,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y,
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return round(
        float(best_threshold),
        2
    )


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

def get_feature_importance(model, X, y):

    estimator = model[-1]

    importance = getattr(
        estimator,
        "feature_importances_",
        None
    )

    if importance is not None:

        return dict(
            zip(
                FEATURES,
                map(float, importance)
            )
        )

    # SVM permutation importance
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc"
    )

    return dict(
        zip(
            FEATURES,
            map(
                float,
                result.importances_mean
            )
        )
    )


# ---------------------------------------------------------
# TRAIN ALL MODELS
# ---------------------------------------------------------

def train_all(names):

    feature_file = PROC / "features.csv"

    if not feature_file.exists():

        raise UserError(
            "Feature dataset missing. "
            "Run 'Build Feature Dataset' first."
        )

    d = pd.read_csv(
        feature_file
    )

    X = d[FEATURES]
    y = d["label"].values

    # -----------------------------------------------------
    # SPATIAL BLOCKS
    # -----------------------------------------------------

    groups = (
        np.floor(
            d.lat / BLOCK_DEG
        ).astype(int).astype(str)
        + "_"
        + np.floor(
            d.lon / BLOCK_DEG
        ).astype(int).astype(str)
    ).values

    unique_groups = len(
        set(groups)
    )

    k = min(
        5,
        unique_groups
    )

    if k < 3:

        raise UserError(
            "Too few spatial blocks for "
            "spatial cross-validation."
        )

    results = {}

    # -----------------------------------------------------
    # MODEL LOOP
    # -----------------------------------------------------

    for name in names:

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"TUNING MODEL: {name}"
        )

        print(
            f"{'=' * 60}"
        )

        best_params = None
        best_auc = -1
        best_oof = None

        grid = list(
            ParameterGrid(
                PARAM_GRIDS[name]
            )
        )

        print(
            f"Parameter combinations: {len(grid)}"
        )

        # -------------------------------------------------
        # HYPERPARAMETER SEARCH
        # -------------------------------------------------

        for i, params in enumerate(grid, 1):

            print(
                f"Testing {i}/{len(grid)}"
            )

            oof = get_oof_predictions(
                name,
                params,
                X,
                y,
                groups,
                k
            )

            try:

                auc = roc_auc_score(
                    y,
                    oof
                )

            except ValueError:

                auc = 0

            if auc > best_auc:

                best_auc = auc
                best_params = params
                best_oof = oof

        print(
            f"\nBest ROC-AUC: {best_auc:.4f}"
        )

        print(
            f"Best parameters: {best_params}"
        )

        # -------------------------------------------------
        # THRESHOLD OPTIMIZATION
        # -------------------------------------------------

        best_threshold = find_best_threshold(
            y,
            best_oof
        )

        print(
            f"Best probability threshold: "
            f"{best_threshold}"
        )

        metrics = compute_metrics(
            y,
            best_oof,
            best_threshold
        )

        # -------------------------------------------------
        # TRAIN FINAL MODEL
        # -------------------------------------------------

        final_model = build(
            name,
            best_params
        )

        final_model.fit(
            X,
            y
        )

        # -------------------------------------------------
        # SAVE MODEL
        # -------------------------------------------------

        joblib.dump(
            final_model,
            MODELS / f"{name}.joblib"
        )

        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        importance = get_feature_importance(
            final_model,
            X,
            y
        )

        # -------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------

        results[name] = {
            **metrics,

            "threshold": best_threshold,

            "best_parameters": best_params,

            "cv": (
                f"GroupKFold({k}) over "
                f"{BLOCK_DEG} degree spatial blocks, "
                f"out-of-fold predictions"
            ),

            "feature_importance": importance
        }

        print(
            "\nFINAL RESULTS"
        )

        print(
            f"Accuracy : {metrics['accuracy']}"
        )

        print(
            f"Precision: {metrics['precision']}"
        )

        print(
            f"Recall   : {metrics['recall']}"
        )

        print(
            f"F1       : {metrics['f1']}"
        )

        print(
            f"ROC-AUC  : {metrics['roc_auc']}"
        )

        print(
            f"Threshold: {best_threshold}"
        )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    (PROC / "metrics.json").write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    return results
if __name__ == "__main__":
    models = [
        "random_forest",
        "svm",
        "xgboost"
    ]

    results = train_all(models)

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    for name, result in results.items():
        print(f"\n{name.upper()}")
        print("-" * 70)
        print(f"Accuracy  : {result['accuracy']}")
        print(f"Precision : {result['precision']}")
        print(f"Recall    : {result['recall']}")
        print(f"F1 Score  : {result['f1']}")
        print(f"ROC-AUC   : {result['roc_auc']}")
        print(f"Threshold : {result['threshold']}")
        print(f"Parameters: {result['best_parameters']}")

    print("\n" + "=" * 70)
    print("Training completed.")
    print("Metrics saved to: data/processed/metrics.json")
    print("=" * 70)