# quanta.py (context-aware)
from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

def _safe_norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0: return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + 1e-12)

def quanta_rank(
    df: pd.DataFrame,
    interactions: List[Dict[str, Any]],
    iid2idx: Dict[str, int],
    item_embs: np.ndarray,
    global_events: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.zeros((0,), dtype=float)

    liked_ids = [x.get("item_id") for x in interactions if x.get("action") in ("like","bag")]
    liked_dom = set(df[df["item_id"].isin(liked_ids)]["domain"].tolist())
    doms = df["domain"].astype(str).tolist()
    novelty = np.array([1.0 if (d not in liked_dom or len(liked_dom)==0) else 0.35 for d in doms], dtype=float)

    ids = df["item_id"].astype(str).tolist()
    idx = [iid2idx[i] for i in ids if i in iid2idx]
    emb = item_embs[idx] if idx else np.zeros((0, item_embs.shape[1]), dtype=float)
    if emb.shape[0] >= 2:
        X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        S = X @ X.T
        crowded = np.asarray(S.mean(axis=1)).reshape(-1)
        crowded = _safe_norm(crowded)
        diversity = 1.0 - crowded
    else:
        diversity = np.ones((len(ids),), dtype=float) * 0.5

    recent = np.zeros((n,), dtype=float)
    if global_events:
        latest_ts = {}
        cand = set(ids)
        for e in global_events:
            it = str(e.get("item_id", ""))
            if it in cand and e.get("action") == "like":
                ts = e.get("ts", 0)
                try:
                    if isinstance(ts, str):
                        from datetime import datetime
                        val = datetime.fromisoformat(ts.replace("Z","")).timestamp()
                    else:
                        val = float(ts)
                except Exception:
                    val = 0.0
                if val > latest_ts.get(it, 0.0):
                    latest_ts[it] = val
        recent = np.array([latest_ts.get(i, 0.0) for i in ids], dtype=float)
        recent = _safe_norm(recent)

    vc = pd.Series(doms).value_counts(normalize=True).to_dict()
    domain_balance = np.array([1.0 - vc.get(d, 0.0) for d in doms], dtype=float)

    def _has_tag(series, tag):
        if tag is None or tag == "": return np.zeros((n,), dtype=float)
        vals = series.fillna("").astype(str).str.lower()
        tag = str(tag).lower()
        return vals.apply(lambda s: 1.0 if tag in s else 0.0).to_numpy(dtype=float)

    region_b = np.zeros((n,), dtype=float)
    festival_b = np.zeros((n,), dtype=float)
    climate_b = np.zeros((n,), dtype=float)
    if context:
        if "region" in df.columns:
            region_b = _has_tag(df["region"], context.get("region"))
        elif "region_tags" in df.columns:
            region_b = _has_tag(df["region_tags"], context.get("region"))

        if "festival" in df.columns:
            festival_b = _has_tag(df["festival"], context.get("festival"))
        elif "festival_tags" in df.columns:
            festival_b = _has_tag(df["festival_tags"], context.get("festival"))

        if "climate" in df.columns:
            climate_b = _has_tag(df["climate"], context.get("climate"))
        elif "climate_tags" in df.columns:
            climate_b = _has_tag(df["climate_tags"], context.get("climate"))

    context_boost = 0.5*region_b + 0.3*festival_b + 0.2*climate_b
    context_boost = _safe_norm(context_boost)

    if "price" in df.columns:
        p = pd.to_numeric(df["price"], errors="coerce")
        med = float(p.median()) if p.notna().any() else 0.0
        pz = (p.fillna(med) - med).abs()
        price = 1.0 - _safe_norm(pz)
    else:
        price = np.ones((n,), dtype=float) * 0.5

    w_nov, w_div, w_rec, w_bal, w_ctx, w_pri = 0.25, 0.22, 0.12, 0.14, 0.17, 0.10
    quanta = (w_nov*novelty + w_div*diversity + w_rec*recent +
              w_bal*domain_balance + w_ctx*context_boost + w_pri*price)

    return _safe_norm(quanta)
