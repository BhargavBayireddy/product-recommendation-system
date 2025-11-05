# gnn_infer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

def _load_artifacts(art_dir: Path):
    emb = art_dir / "item_embeddings.npy"
    mps = art_dir / "idmaps.json"
    if not (emb.exists() and mps.exists()):
        return None, None
    try:
        import json
        embs = np.load(emb)
        maps = json.loads(mps.read_text(encoding="utf-8"))
        iid2idx = {str(k): int(v) for k,v in maps.get("iid2idx", {}).items()}
        return embs.astype(np.float32), iid2idx
    except Exception:
        return None, None

def load_item_embeddings(items: pd.DataFrame, artifacts_dir: Path) -> Tuple[np.ndarray, Dict[str,int], str]:
    trained, trained_i2i = _load_artifacts(artifacts_dir)
    current_ids = items["item_id"].astype(str).tolist()
    if trained is not None and trained_i2i:
        if all(i in trained_i2i for i in current_ids):
            idx = np.array([trained_i2i[i] for i in current_ids], dtype=int)
            return trained[idx], {i:j for j,i in enumerate(current_ids)}, "LightGCN (GNN)"
    # fallback
    rng = np.random.default_rng(123)
    embs = rng.normal(0, 0.1, size=(len(current_ids), 32)).astype(np.float32)
    return embs, {i:j for j,i in enumerate(current_ids)}, "RandomFallback"

def make_user_vector(interactions, iid2idx: Dict[str,int], item_embs: np.ndarray) -> np.ndarray:
    liked = [x.get("item_id") for x in interactions if x.get("action") in ("like","bag")]
    idx = [iid2idx[i] for i in liked if i in iid2idx]
    if idx:
        return item_embs[idx].mean(axis=0, keepdims=True)
    return item_embs.mean(axis=0, keepdims=True)
