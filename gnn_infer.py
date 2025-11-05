# gnn_infer.py
"""
LightGCN inference helpers (no training here).
- If artifacts with real embeddings exist, use them (BACKEND="LightGCN").
- Else, fall back to random embeddings aligned to current items (BACKEND="RandomFallback").
Artifacts expected (inside ./artifacts):
  - item_embeddings.npy  : shape [N_items_trained, D]
  - idmaps.json          : {"iid2idx": {"item_id": index, ...}}
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

def _load_artifacts(artifacts_dir: Path):
    emb_path = artifacts_dir / "item_embeddings.npy"
    map_path = artifacts_dir / "idmaps.json"
    if not (emb_path.exists() and map_path.exists()):
        return None, None
    try:
        embs = np.load(emb_path)
        maps = json.loads(map_path.read_text(encoding="utf-8"))
        iid2idx = {str(k): int(v) for k, v in maps.get("iid2idx", {}).items()}
        return embs.astype(np.float32), iid2idx
    except Exception:
        return None, None

def load_item_embeddings(items: pd.DataFrame, artifacts_dir: Path) -> Tuple[np.ndarray, Dict[str,int], str]:
    """Return (ITEM_EMBS aligned-to-items, iid2idx_for_current_items, BACKEND string)."""
    trained_embs, trained_i2i = _load_artifacts(artifacts_dir)

    current_ids: List[str] = items["item_id"].astype(str).tolist()

    if trained_embs is not None and trained_i2i:
        # Can we align all current items to trained ids?
        missing = [it for it in current_ids if it not in trained_i2i]
        if len(missing) == 0:
            # Perfect alignment: reorder to match current ITEMS
            order_idx = np.array([trained_i2i[it] for it in current_ids], dtype=int)
            embs_cur = trained_embs[order_idx]
            iid2idx_cur = {it: i for i, it in enumerate(current_ids)}
            return embs_cur, iid2idx_cur, "LightGCN (GNN)"

    # Fallback: random (still deterministic)
    rng = np.random.default_rng(123)
    embs = rng.normal(0, 0.1, size=(len(current_ids), 32)).astype(np.float32)
    iid2idx_cur = {it: i for i, it in enumerate(current_ids)}
    return embs, iid2idx_cur, "RandomFallback"

def make_user_vector(interactions, iid2idx: Dict[str,int], item_embs: np.ndarray) -> np.ndarray:
    """Average of liked/bagged item vectors; else global mean."""
    liked = [x.get("item_id") for x in interactions if x.get("action") in ("like", "bag")]
    idx = [iid2idx[i] for i in liked if i in iid2idx]
    if idx:
        return item_embs[idx].mean(axis=0, keepdims=True)
    return item_embs.mean(axis=0, keepdims=True)
