# recommender.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path

from gnn_infer import load_item_embeddings, make_user_vector

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
    return v / n

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (_normalize(a) @ _normalize(b).T)

def build_catalog(items_df: pd.DataFrame) -> pd.DataFrame:
    df = items_df.copy()
    # Ensure columns
    for col in ["item_id","name","domain","category","mood"]:
        if col not in df.columns: df[col] = ""
    df["query_blob"] = (df["name"].astype(str) + " " + df["domain"].astype(str) +
                        " " + df["category"].astype(str) + " " + df["mood"].astype(str)).str.lower()
    return df

def search_live(df: pd.DataFrame, q: str, limit: int = 60) -> pd.DataFrame:
    if not q: return df.head(0)
    q = q.strip().lower()
    # simple fast contains; if many items consider fuzzy
    m = df["query_blob"].str.contains(q, na=False)
    out = df[m].copy()
    # light score: startswith > contains
    out["_s"] = (out["name"].str.lower().str.startswith(q)).astype(int) * 2 + 1
    return out.sort_values(["_s","name"], ascending=[False,True]).head(limit).drop(columns=["_s"])

def hybrid_recommend(
    items_df: pd.DataFrame,
    interactions_user: List[Dict],
    interactions_global: List[Dict],
    artifacts_dir: Path,
    k_return: int = 48
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Returns (top, collab, explore, backend_name)
    """
    items = build_catalog(items_df)
    # --- Embeddings (GNN if available else random fallback) ---
    emb_mat, iid2idx, backend = load_item_embeddings(items, artifacts_dir)

    # User vector from likes/bag
    uvec = make_user_vector(interactions_user, iid2idx, emb_mat)   # [1, D]
    scores_content = _cosine(uvec, emb_mat)[0]                     # [N]

    # Collaborative: items liked by similar users (global)
    # Build simple item popularity weighted by "similarity" using overlap on user's likes
    user_likes = {x["item_id"] for x in interactions_user if x.get("action") in ("like","bag")}
    pop: Dict[str, float] = {}
    for row in interactions_global:
        iid = row.get("item_id")
        act = row.get("action")
        if not iid or act not in ("like","bag"): continue
        # small boost if this item intersects with user likes history domain/category
        boost = 1.0 + 0.3 * int(iid in user_likes)
        pop[iid] = pop.get(iid, 0.0) + boost

    collab_scores = np.zeros(len(items), dtype=np.float32)
    for iid, sc in pop.items():
        if iid in iid2idx:
            collab_scores[iid2idx[iid]] += sc

    # Blend
    alpha = 0.65  # content weight
    blend = alpha * scores_content + (1 - alpha) * (collab_scores / (collab_scores.max() + 1e-8))

    # Exclude already strongly interacted items for top section
    seen = set(user_likes)
    order = np.argsort(-blend)
    top_idx = [i for i in order if items.iloc[i]["item_id"] not in seen][:k_return]
    top = items.iloc[top_idx].copy()

    # Collab section: highest collaborative not already in top
    collab_order = np.argsort(-collab_scores)
    collab_idx = [i for i in collab_order if items.iloc[i]["item_id"] not in seen][:k_return]
    collab_df = items.iloc[collab_idx].copy()

    # Explore: low-similarity/randomized tail
    tail = items.iloc[order[-(k_return*3):]].sample(min(k_return, len(items)), random_state=7)
    explore = tail.copy()

    return top, collab_df, explore, backend
