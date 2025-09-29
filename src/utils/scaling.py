from typing import List, Dict, Set, Tuple

def minmax_scale(scores: Dict[str, float], scale_min=0.0, scale_max=10.0) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return {k: scale_min for k in scores}  # 모든 점수가 같으면 최소값으로 고정

    return {
        k: scale_min + (v - min_val) / (max_val - min_val) * (scale_max - scale_min)
        for k, v in scores.items()
    }