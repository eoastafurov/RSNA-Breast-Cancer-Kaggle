import numpy as np
from typing import Optional


def c_metrics(probs, y_test, eps: Optional[float] = 1e-8):
    """Сalculation of metrics in vectorized format.
    Robust to batch without positive classes (y_true_count
     == 0)

    Args:
        probs (np.ndarray): flat numpy NDArray containing
            probabilities of positive class.
        y_test (np.ndarray): flat numpy NDArray containing
            ground truth labels.
        eps (float, optional): machine epsilon. Defaults to
            1e-8.

    Returns:
        Dict[str, float]: continious classification metrics
    """
    ctp = np.dot(probs, y_test.astype(bool))
    cfp = np.sum(probs) - ctp
    c_precision = ctp / (ctp + cfp + eps)
    c_recall = ctp / (np.sum(y_test) + eps)
    cf1 = 2 * (c_precision * c_recall) / (c_precision + c_recall + eps)
    return {
        "ctp": ctp,
        "cfp": cfp,
        "c_recall": c_recall,
        "c_precision": c_precision,
        "cf1": cf1,
    }


def pfbeta_torch(preds, labels, beta=1):
    y_true_count = labels.sum()
    ctp = preds[labels == 1].sum()
    cfp = preds[labels == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall)
        )
        return result
    else:
        return 0.0


def pfbeta_torch_thresh(preds, labels):
    optimized_preds, threshold = optimize_preds(preds, labels)
    return pfbeta_torch(optimized_preds, labels), threshold


def optimize_preds(
    preds, labels, thresh=None, return_thresh=False, print_results=False
):
    preds = preds.clone()
    without_thresh = pfbeta_torch(preds, labels)

    if not thresh and labels is not None:
        threshs = np.linspace(0, 1, 1001)
        f1s = [pfbeta_torch((preds > thr).float(), labels) for thr in threshs]
        idx = np.argmax(f1s)
        thresh, best_pfbeta = threshs[idx], f1s[idx]

    preds = (preds > thresh).float()

    if print_results:
        print(f"without optimization: {without_thresh}")
        pfbeta = pfbeta_torch(preds, labels)
        print(f"with optimization: {pfbeta}")
        print(f"best_thresh = {thresh}")
    if return_thresh:
        return thresh
    return preds, thresh
