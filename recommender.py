# recommender.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from gnn_infer import load_item_embeddings, make_user_vector

def prepare_items(items_df: pd.DataFrame) -> pd.DataFrame:
    items = items_df.copy()
    # must have: item_id, title, domain, provider columns
    for col, default in [("domain","Entertainment"), ("provider","")]:
        if col not in items.columns:
            items[col] = default
    return items

def recommend_items(
    items: pd.DataFrame,
    my_interactions: List[Dict],
    global_interactions: List[Dict],
    artifacts_dir: Path,
    k_top: int = 48
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    items = prepare_items(items)
    embs, iid2idx, backend = load_item_embeddings(items, artifacts_dir)
    uvec = make_user_vector(my_interactions, iid2idx, embs)  # [1, D]
    sims = cosine_similarity(uvec, embs)[0]  # [N]
    items = items.copy()
    items["score_gnn"] = sims

    # hide already interacted
    seen_ids = {str(x["item_id"]) for x in my_interactions}
    fresh = items[~items["item_id"].astype(str).isin(seen_ids)]

    # ---- Top picks (personal) ----
    top = (fresh.sort_values("score_gnn", ascending=False)
                .head(k_top))

    # ---- Collab (vibe-twins) ----
    # Build item -> users and user -> items quickly
    g = pd.DataFrame(global_interactions) if global_interactions else pd.DataFrame(columns=["uid","item_id","action"])
    collab = pd.DataFrame(columns=items.columns)
    if not g.empty:
        g = g[g["action"].isin(["like","bag"])]
        # users who touched anything I touched
        my_seen = {str(x["item_id"]) for x in my_interactions if x["action"] in ("like","bag")}
        if my_seen:
            overlap_users = set(g[g["item_id"].astype(str).isin(my_seen)]["uid"])
            others = g[g["uid"].isin(overlap_users)]
            candidate_ids = set(others["item_id"].astype(str)) - my_seen
            collab = items[items["item_id"].astype(str).isin(candidate_ids)].copy()
            # rank by popularity among overlap users then gnn
            pop = others.groupby("item_id").size().rename("pop")
            collab = collab.merge(pop, left_on="item_id", right_index=True, how="left").fillna({"pop":0})
            collab["score_collab"] = collab["pop"].astype(float) + 0.3*collab["score_gnn"]
            collab = collab.sort_values(["score_collab","score_gnn"], ascending=False).head(k_top)

    # ---- Explore (novel) ----
    # lowest popularity but high gnn among unseen domains
    def novelty_rank(df):
        # simple: inverse domain frequency
        freq = df["domain"].value_counts().to_dict()
        return df["score_gnn"] * df["domain"].map(lambda d: 1.0/(1+freq.get(d,1)))
    explore = fresh.copy()
    explore["novel"] = novelty_rank(explore)
    explore = explore.sort_values(["novel","score_gnn"], ascending=False).head(k_top)

    return top, collab, explore, backend
