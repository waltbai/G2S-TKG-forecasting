from typing import List, Dict

from src.utils.common import format_params


def compute_rank(
        preds: List[str],
        answer: str,
        filters: List[str] = None
) -> int:
    """Compute rank of each true label."""
    rank = None
    try:
        if filters is not None:
            preds = [_ for _ in preds if _ not in filters]
        rank = preds.index(answer)
    except ValueError:
        pass
    return rank


def compute_metrics(
        tot_preds: List[List[str]],
        tot_answers: List[str],
        tot_filters: List[List[str]],
) -> Dict[str, float]:
    """Compute metrics.

    Since MRR is not applicable, we only compute Hit@k.
    """
    raw_hit1, raw_hit3, raw_hit10 = 0, 0, 0
    filter_hit1, filter_hit3, filter_hit10 = 0, 0, 0
    total = 0
    for preds, answer, filters in zip(tot_preds, tot_answers, tot_filters):
        raw_rank = compute_rank(preds, answer)
        rank = compute_rank(preds, answer, filters)
        if raw_rank is not None:
            if raw_rank < 1:
                raw_hit1 += 1
            if raw_rank < 3:
                raw_hit3 += 1
            if raw_rank < 10:
                raw_hit10 += 1
        if rank is not None:
            if rank < 1:
                filter_hit1 += 1
            if rank < 3:
                filter_hit3 += 1
            if rank < 10:
                filter_hit10 += 1
        total += 1
    return {
        "raw hit@1": raw_hit1 / total,
        "raw hit@3": raw_hit3 / total,
        "raw hit@10": raw_hit10 / total,
        "filter hit@1": filter_hit1 / total,
        "filter hit@3": filter_hit3 / total,
        "filter hit@10": filter_hit10 / total,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics into string."""
    params = [
        ("raw hit@1", f"{metrics['raw hit@1']:.2%}"),
        ("raw hit@3", f"{metrics['raw hit@3']:.2%}"),
        ("raw hit@10", f"{metrics['raw hit@10']:.2%}"),
        ("filter hit@1", f"{metrics['filter hit@1']:.2%}"),
        ("filter hit@3", f"{metrics['filter hit@3']:.2%}"),
        ("filter hit@10", f"{metrics['filter hit@10']:.2%}"),
    ]
    return format_params(params)
