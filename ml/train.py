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


# =========================================================
# MODEL BUILDERS
# =========================================================

def build(name, params=None):

    params = params or {}

    # -----------------------------------------------------
    # RANDOM FOREST
    # -----------------------------------------------------

    if name == "random_forest":

        return make_pipeline(
            SimpleImputer(strategy="median"),

            RandomForestClassifier(
                n_estimators=params.get(
                    "n_estimators", 300
                ),

                max_depth=params.get(
                    "max_depth", None
                ),

                min_samples_split=params.get(
                    "min_samples_split", 2
                ),

                min_samples_leaf=params.get(
                    "min_samples_leaf", 1
                ),

                max_features=params.get(
                    "max_features", "sqrt"
                ),

                class_weight="balanced",

                random_state=42,

                n_jobs=-1,
            )
        )

    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

    if name == "svm":

        return make_pipeline(
            SimpleImputer(strategy="median"),

            StandardScaler(),

            SVC(
                C=params.get(
                    "C", 1.0
                ),

                gamma=params.get(
                    "gamma", "scale"
                ),

                kernel=params.get(
                    "kernel", "rbf"
                ),

                probability=True,

                class_weight="balanced",

                random_state=42,
            )
        )

    # -----------------------------------------------------
    # XGBOOST
    # -----------------------------------------------------

    if name == "xgboost":

        try:
            from xgboost import XGBClassifier

        except ImportError:

            raise UserError(
                "XGBoost not installed. "
                "Run: pip install xgboost"
            )

        return make_pipeline(
            SimpleImputer(strategy="median"),

            XGBClassifier(

                n_estimators=params.get(
                    "n_estimators", 300
                ),

                max_depth=params.get(
                    "max_depth", 4
                ),

                learning_rate=params.get(
                    "learning_rate", 0.05
                ),

                min_child_weight=params.get(
                    "min_child_weight", 1
                ),

                subsample=params.get(
                    "subsample", 0.9
                ),

                colsample_bytree=params.get(
                    "colsample_bytree", 0.9
                ),

                gamma=params.get(
                    "gamma", 0
                ),

                reg_alpha=params.get(
                    "reg_alpha", 0
                ),

                reg_lambda=params.get(
                    "reg_lambda", 1
                ),

                scale_pos_weight=params.get(
                    "scale_pos_weight", 1
                ),

                eval_metric="logloss",

                random_state=42,

                n_jobs=-1,
            )
        )

    raise UserError(
        f"Unknown model '{name}'."
    )


# =========================================================
# SMALL / TARGETED PARAMETER GRIDS
#
# IMPORTANT:
# These grids are intentionally small.
# The previous XGBoost search had 7776 combinations.
#
# New total:
# Random Forest = 24
# SVM           = 12
# XGBoost       = 32
# Total         = 68
# =========================================================

PARAM_GRIDS = {

    # -----------------------------------------------------
    # RANDOM FOREST = 24
    # -----------------------------------------------------

    "random_forest": {

        "n_estimators": [
            300,
            500
        ],

        "max_depth": [
            None,
            10
        ],

        "min_samples_split": [
            2,
            5
        ],

        "min_samples_leaf": [
            1,
            2
        ],

        "max_features": [
            "sqrt"
        ],
    },


    # -----------------------------------------------------
    # SVM = 12
    # -----------------------------------------------------

    "svm": {

        "C": [
            0.1,
            1,
            10
        ],

        "gamma": [
            "scale",
            0.01
        ],

        "kernel": [
            "rbf"
        ],
    },


    # -----------------------------------------------------
    # XGBOOST = 32
    # -----------------------------------------------------

    "xgboost": {

        "n_estimators": [
            200,
            300
        ],

        "max_depth": [
            3,
            4
        ],

        "learning_rate": [
            0.05,
            0.08
        ],

        "min_child_weight": [
            1,
            5
        ],

        "subsample": [
            0.8,
            1.0
        ],

        "colsample_bytree": [
            0.8,
            1.0
        ],

        "scale_pos_weight": [
            1,
            2.5
        ],
    },
}


# =========================================================
# METRICS
# =========================================================

def compute_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {

        "accuracy": round(
            float(
                accuracy_score(
                    y_true,
                    predictions
                )
            ),
            4
        ),

        "precision": round(
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),
            4
        ),

        "recall": round(
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),
            4
        ),

        "f1": round(
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0
                )
            ),
            4
        ),

        "roc_auc": round(
            float(
                roc_auc_score(
                    y_true,
                    probabilities
                )
            ),
            4
        ),

        "confusion": [
            [
                int(tn),
                int(fp)
            ],
            [
                int(fn),
                int(tp)
            ]
        ]
    }


# =========================================================
# OOF PREDICTIONS
# =========================================================

def get_oof_predictions(
    model_name,
    params,
    X,
    y,
    groups,
    k
):

    oof = np.zeros(
        len(X),
        dtype=float
    )

    cv = GroupKFold(
        n_splits=k
    )

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

        probabilities = (
            model.predict_proba(
                X.iloc[test_idx]
            )[:, 1]
        )

        oof[test_idx] = probabilities

    return oof


# =========================================================
# THRESHOLD OPTIMIZATION
# =========================================================

def find_best_threshold(
    y,
    probabilities
):

    best_threshold = 0.5
    best_f1 = -1

    # -----------------------------------------------------
    # Threshold range
    #
    # Lower thresholds are useful here because the dataset
    # is imbalanced and we want reasonable positive recall.
    # -----------------------------------------------------

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


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(
    model,
    X,
    y
):

    estimator = model[-1]

    importance = getattr(
        estimator,
        "feature_importances_",
        None
    )

    # -----------------------------------------------------
    # Tree models
    # -----------------------------------------------------

    if importance is not None:

        return dict(
            zip(
                FEATURES,
                map(
                    float,
                    importance
                )
            )
        )

    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

    from sklearn.inspection import (
        permutation_importance
    )

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


# =========================================================
# TRAIN ALL MODELS
# =========================================================

def train_all(names):

    feature_file = (
        PROC / "features.csv"
    )

    if not feature_file.exists():

        raise UserError(
            "Feature dataset missing. "
            "Run 'Build Feature Dataset' first."
        )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    d = pd.read_csv(
        feature_file
    )

    X = d[FEATURES]

    y = d["label"].values

    # -----------------------------------------------------
    # DISPLAY DATASET INFORMATION
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(
        f"Total samples : {len(d)}"
    )

    print(
        f"Positive      : {int(y.sum())}"
    )

    print(
        f"Background    : {int((y == 0).sum())}"
    )

    print(
        f"Features      : {len(FEATURES)}"
    )

    # -----------------------------------------------------
    # SPATIAL BLOCKS
    # -----------------------------------------------------

    groups = (

        np.floor(
            d.lat / BLOCK_DEG
        )
        .astype(int)
        .astype(str)

        + "_"

        + np.floor(
            d.lon / BLOCK_DEG
        )
        .astype(int)
        .astype(str)

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

    print(
        f"Spatial blocks : {unique_groups}"
    )

    print(
        f"CV folds       : {k}"
    )

    print(
        f"Block size     : {BLOCK_DEG} degrees"
    )

    print("=" * 70)

    results = {}

    # =====================================================
    # MODEL LOOP
    # =====================================================

    for name in names:

        print()
        print("=" * 60)
        print(
            f"TUNING MODEL: {name.upper()}"
        )
        print("=" * 60)

        # -------------------------------------------------
        # SEARCH SPACE
        # -------------------------------------------------

        grid = list(
            ParameterGrid(
                PARAM_GRIDS[name]
            )
        )

        print(
            f"Parameter combinations: {len(grid)}"
        )

        best_params = None
        best_auc = -1
        best_oof = None

        # -------------------------------------------------
        # HYPERPARAMETER SEARCH
        # -------------------------------------------------

        for i, params in enumerate(
            grid,
            1
        ):

            print(
                f"Testing {i}/{len(grid)}",
                flush=True
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

                auc = 0.0

            if auc > best_auc:

                best_auc = auc

                best_params = params

                best_oof = oof

        # -------------------------------------------------
        # BEST PARAMETERS
        # -------------------------------------------------

        print()

        print(
            f"Best ROC-AUC: {best_auc:.4f}"
        )

        print(
            f"Best parameters: {best_params}"
        )

        # -------------------------------------------------
        # THRESHOLD
        # -------------------------------------------------

        best_threshold = (
            find_best_threshold(
                y,
                best_oof
            )
        )

        print(
            f"Best probability threshold: "
            f"{best_threshold}"
        )

        # -------------------------------------------------
        # FINAL OOF METRICS
        # -------------------------------------------------

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

        model_path = (
            MODELS / f"{name}.joblib"
        )

        joblib.dump(
            final_model,
            model_path
        )

        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        importance = (
            get_feature_importance(
                final_model,
                X,
                y
            )
        )

        # -------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------

        results[name] = {

            **metrics,

            "threshold":
                best_threshold,

            "best_parameters":
                best_params,

            "cv":
                (
                    f"GroupKFold({k}) over "
                    f"{BLOCK_DEG} degree spatial blocks, "
                    f"out-of-fold predictions"
                ),

            "feature_importance":
                importance
        }

        # -------------------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------------------

        print()
        print(
            "FINAL RESULTS"
        )

        print(
            f"Accuracy : "
            f"{metrics['accuracy']}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']}"
        )

        print(
            f"Recall   : "
            f"{metrics['recall']}"
        )

        print(
            f"F1       : "
            f"{metrics['f1']}"
        )

        print(
            f"ROC-AUC  : "
            f"{metrics['roc_auc']}"
        )

        print(
            f"Threshold: "
            f"{best_threshold}"
        )

        print(
            f"Confusion: "
            f"{metrics['confusion']}"
        )

        print(
            f"Model saved: "
            f"{model_path}"
        )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics_file = (
        PROC / "metrics.json"
    )

    metrics_file.write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    return results


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    models = [

        "random_forest",

        "svm",

        "xgboost"
    ]

    results = train_all(
        models
    )

    # =====================================================
    # FINAL MODEL COMPARISON
    # =====================================================

    print()
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    for name, result in results.items():

        print()
        print(
            name.upper()
        )

        print(
            "-" * 70
        )

        print(
            f"Accuracy  : "
            f"{result['accuracy']}"
        )

        print(
            f"Precision : "
            f"{result['precision']}"
        )

        print(
            f"Recall    : "
            f"{result['recall']}"
        )

        print(
            f"F1 Score  : "
            f"{result['f1']}"
        )

        print(
            f"ROC-AUC   : "
            f"{result['roc_auc']}"
        )

        print(
            f"Threshold : "
            f"{result['threshold']}"
        )

        print(
            f"Confusion : "
            f"{result['confusion']}"
        )

        print(
            f"Parameters: "
            f"{result['best_parameters']}"
        )

    print()
    print("=" * 70)
    print(
        "Training completed."
    )
    print(
        "Metrics saved to: "
        "data/processed/metrics.json"
    )
    print("=" * 70)