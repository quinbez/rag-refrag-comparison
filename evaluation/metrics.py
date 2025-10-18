import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def exact_match(pred, ground_truths):
    """1 if any ground truth exactly matches the prediction"""
    pred_norm = normalize_answer(pred)
    return int(any(pred_norm == normalize_answer(gt) for gt in ground_truths))

def f1_score(pred, ground_truths):
    """Compute max F1 over all ground truth answers"""
    pred_tokens = normalize_answer(pred).split()
    f1_scores = []
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            f1_scores.append(0.0)
            continue
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1_scores.append(2 * precision * recall / (precision + recall))
    return max(f1_scores)
