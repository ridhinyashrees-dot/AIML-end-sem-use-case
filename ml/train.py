import json, joblib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from backend.config import PROC, MODELS, FEATURES, BLOCK_DEG, UserError
from ml.evaluate import compute

def build(name):
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(300, class_weight="balanced", random_state=42, n_jobs=-1))
    if name == "svm":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=42))
    if name == "xgboost":
        try: from xgboost import XGBClassifier
        except ImportError: raise UserError("XGBoost not installed. Run: pip install xgboost")
        return make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42))
    raise UserError(f"Unknown model '{name}'.")

def train_all(names):
    f = PROC / "features.csv"
    if not f.exists(): raise UserError("Feature dataset missing. Run 'Build Feature Dataset' first.")
    d = pd.read_csv(f); X, y = d[FEATURES], d["label"].values
    # Spatial block validation: points in the same BLOCK_DEG x BLOCK_DEG block never split across train/test.
    groups = (np.floor(d.lat / BLOCK_DEG).astype(int).astype(str) + "_" + np.floor(d.lon / BLOCK_DEG).astype(int).astype(str)).values
    k = min(5, len(set(groups)))
    if k < 3: raise UserError("Too few spatial blocks for spatial cross-validation; study area too small.")
    results = {}
    for n in names:
        oof = np.zeros(len(d))
        for tr, te in GroupKFold(k).split(X, y, groups):
            m = build(n).fit(X.iloc[tr], y[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        m = build(n).fit(X, y); est = m[-1]
        imp = getattr(est, "feature_importances_", None)
        if imp is None:   # SVM: permutation importance on the training data
            from sklearn.inspection import permutation_importance
            imp = permutation_importance(m, X, y, n_repeats=3, random_state=42, scoring="roc_auc").importances_mean
        joblib.dump(m, MODELS / f"{n}.joblib")
        results[n] = {**compute(y, oof), "cv": f"GroupKFold({k}) over {BLOCK_DEG}° spatial blocks, out-of-fold predictions",
                      "feature_importance": dict(zip(FEATURES, map(float, imp)))}
    (PROC / "metrics.json").write_text(json.dumps(results, indent=2))
    return results
