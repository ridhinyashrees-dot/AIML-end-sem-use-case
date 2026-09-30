import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def compute(y, proba, thr=0.5):
    p = (proba >= thr).astype(int)
    m = {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
         "recall": recall_score(y, p, zero_division=0), "f1": f1_score(y, p, zero_division=0),
         "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist()}
    try: m["roc_auc"] = roc_auc_score(y, proba)
    except ValueError: m["roc_auc"] = None
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}
