import numpy as np
import rlevaluation._utils.numba as nbu

from numba.typed import List as NList
from typing import Any, NamedTuple, Tuple

from rlevaluation.backend.statistics import stratified_percentile_bootstrap_ci
from rlevaluation.hypers.utils import idx_preference


class BootstrapHyperResult(NamedTuple):
    best_idx: int
    best_score: float

    uncertainty_set_idxs: np.ndarray
    uncertainty_set_probs: np.ndarray
    sample_stat: float
    ci: Tuple[float, float]


@nbu.njit
def bootstrap_hyper_selection(
    rng: np.random.Generator,
    score_per_seed: NList[np.ndarray],
    statistic: Any,
    prefer: str,
    threshold: float = 0.05,
    iterations: int = 1000,
) -> BootstrapHyperResult:
    n_hypers = len(score_per_seed)

    out = np.empty(iterations, dtype=np.int64)
    out_scores = np.empty(iterations, dtype=np.float64)

    # simulate many possible hyper-search outcomes
    for i in range(iterations):
        sim_scores = np.empty(n_hypers, dtype=np.float64)

        for h in range(n_hypers):
            hyper_scores = score_per_seed[h]
            idxs = rng.integers(0, len(hyper_scores), size=len(hyper_scores))

            sim_scores[h] = statistic(hyper_scores[idxs])

        best_h = idx_preference(prefer, sim_scores)
        out[i] = best_h
        out_scores[i] = sim_scores[best_h]

    # calculate sample scores
    scores = np.empty(n_hypers, dtype=np.float64)
    for h in range(n_hypers):
        scores[h] = statistic(score_per_seed[h])

    best_idx: Any = idx_preference(prefer, scores)
    best_score: Any = scores[best_idx]

    # determine which indices could be the best
    all_possible_idxs = np.unique(out)
    unc_set = []
    unc_probs = []
    unc_scores = []
    for i in range(len(all_possible_idxs)):
        idx = all_possible_idxs[i]
        n = np.sum(out == idx)

        p = n / iterations
        if p >= threshold:
            unc_set.append(idx)
            unc_probs.append(p)
            unc_scores.append(score_per_seed[idx])

    unc_set_idxs = np.array(unc_set, dtype=np.int64)
    unc_set_probs = np.array(unc_probs, dtype=np.float64)

    bs_res = stratified_percentile_bootstrap_ci(
        rng,
        unc_scores,
        unc_set_probs,
        statistic,
        threshold,
    )

    return BootstrapHyperResult(
        best_idx=best_idx,
        best_score=best_score,
        uncertainty_set_idxs=unc_set_idxs,
        uncertainty_set_probs=unc_set_probs,
        sample_stat=bs_res.sample_stat,
        ci=bs_res.ci,
    )
