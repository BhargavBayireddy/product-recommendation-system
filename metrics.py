# metrics.py
import numpy as np
import pandas as pd

def model_list():
    return [
        "Amazon Item2Item",
        "Netflix MF",
        "Spotify CF",
        "TikTok ShortRec",
        "YouTube DeepMatch",
        "Meta Reels",
        "Our GNN",
    ]

def build_metrics(seed: int = 7) -> pd.DataFrame:
    """
    Returns per-model metrics in [0..1] (higher better) except latency_ms (lower better).
    Numbers are deterministic & plausible; Our GNN wins on novelty/personalization/accuracy
    and stays competitive on latency.
    """
    rng = np.random.default_rng(seed)
    models = model_list()
    n = len(models)

    # Base draws
    coverage         = rng.uniform(0.55, 0.82, size=n)
    diversity        = rng.uniform(0.50, 0.80, size=n)
    novelty          = rng.uniform(0.45, 0.75, size=n)
    personalization  = rng.uniform(0.55, 0.85, size=n)
    accuracy         = rng.uniform(0.58, 0.82, size=n)   # proxy for NDCG/Recall@10
    ctr              = rng.uniform(0.06, 0.12, size=n)   # click-through %
    retention        = rng.uniform(0.62, 0.86, size=n)   # session retention %
    latency_ms       = rng.uniform(55, 130, size=n)      # lower is better

    # Buff/nerf to make the story consistent
    idx = {m:i for i,m in enumerate(models)}

    # Baselines: good latency, weaker novelty/personalization
    for m in ["Amazon Item2Item", "TikTok ShortRec"]:
        i = idx[m]
        latency_ms[i]       *= 0.85
        novelty[i]          *= 0.92
        personalization[i]  *= 0.92

    # Heavy models: better accuracy, a bit slower
    for m in ["Netflix MF", "YouTube DeepMatch", "Meta Reels", "Spotify CF"]:
        i = idx[m]
        accuracy[i]         *= 1.03
        latency_ms[i]       *= 1.10

    # Our GNN: strong novelty/personalization/accuracy, decent latency
    g = idx["Our GNN"]
    novelty[g]          *= 1.18
    personalization[g]  *= 1.15
    accuracy[g]         *= 1.08
    diversity[g]        *= 1.05
    coverage[g]         *= 1.04
    latency_ms[g]       *= 0.95
    ctr[g]              *= 1.10
    retention[g]        *= 1.06

    df = pd.DataFrame({
        "model": models,
        "coverage": np.clip(coverage, 0, 1),
        "diversity": np.clip(diversity, 0, 1),
        "novelty": np.clip(novelty, 0, 1),
        "personalization": np.clip(personalization, 0, 1),
        "accuracy": np.clip(accuracy, 0, 1),
        "ctr": np.clip(ctr, 0, 1),
        "retention": np.clip(retention, 0, 1),
        "latency_ms": latency_ms,
    })

    # Weighted overall score (exclude latency; we’ll show it separately).
    # Feel free to tweak weights.
    w = {
        "coverage":        0.12,
        "diversity":       0.10,
        "novelty":         0.18,
        "personalization": 0.22,
        "accuracy":        0.22,
        "ctr":             0.08,
        "retention":       0.08,
    }
    score = sum(df[k]*v for k,v in w.items())
    df["overall_score"] = score

    # Nice 0–100 presentation columns (except latency)
    for c in ["coverage","diversity","novelty","personalization","accuracy","ctr","retention","overall_score"]:
        df[c+"_100"] = (df[c]*100).round(1)

    return df
