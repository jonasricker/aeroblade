import numpy as np
from sklearn.metrics import roc_curve


def tpr_at_max_fpr(y_true, y_score, max_fpr) -> float:
    """Return the TPR that ensures a certain maximum FPR."""
    fpr, tpr, _ = roc_curve(
        y_true=y_true,
        y_score=y_score,
        drop_intermediate=False,
    )
    index = np.argmax(fpr > max_fpr) - 1  # np.argmax returns index of first True
    return tpr[index]
