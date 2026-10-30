"""Event-detection metrics."""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def metric_summary(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 score": float(f1_score(y_true, y_pred, zero_division=0)),
        "False-alarm rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "Missed-event rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "True negatives": int(tn), "False positives": int(fp),
        "False negatives": int(fn), "True positives": int(tp),
    }
