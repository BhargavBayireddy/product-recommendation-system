# gnn_infer.py — Quanta GNN embeddings, propagation, and recommendation logic
# Loads from /artifacts when available; otherwise builds a synthetic demo catalog.

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable
from pathlib import Path
import numpy as np
import pandas as pd

ART_DIR = Path("artifacts")

class QuantaGNN:
    """
    Minimal placeholder Quanta GNN for node-level propagation.
    """
    def __init__(self, num_items: int, dim: int):
        self.embeddings = np.random.RandomState(7).rand(num_items, dim).astype(np.float32)

    def propagate(self, A: np.ndarray, X: np.ndarray, steps: int = 2) -> np.ndarray:
        # Single-step linear propagation (lightly) with residual
        H = X.copy()
        for _ in range(steps):
            H = H + 0.05 * (A @ H)
        return H

# -------------------- Utilities --------------------

def _normalize_rows(X: np.ndarray, eps=1e-8) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n

def _cosine_scores(u: np.ndarray, X: np.ndarray) -> np.ndarray:
    u = u / (np.linalg.norm(u) + 1e-8)
    Xn = _normalize_rows(X)
    return Xn @ u

def _demo_items() -> pd.DataFrame:
    # keep in sync with app demo
    data = [
        ("mv001", "Interstellar", "Netflix", "Sci-Fi", "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1200", "Space travel father-daughter time warp epic"),
        ("mv002", "Inception", "Netflix", "Thriller", "https://images.unsplash.com/photo-1496302662116-35cc4f36df92?q=80&w=1200", "Dream within dream heist mind-bending"),
        ("pr001", "Sony WH-1000XM5", "Amazon", "Electronics", "https://images.unsplash.com/photo-1518447958130-6f57f3f3b7b0?q=80&w=1200", "Noise cancelling headphones premium audio"),
        ("pr002", "Kindle Paperwhite", "Amazon", "Books", "https://images.unsplash.com/photo-1513475382585-d06e58bcb0ea?q=80&w=1200", "E-reader glare free waterproof"),
        ("mu001", "Blinding Lights", "Spotify", "Synthwave", "https://images.unsplash.com/photo-1511379938547-c1f69419868d?q=80&w=1200", "Electropop retro vibes catchy chorus"),
        ("mu002", "Shape of You", "Spotify", "Pop", "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?q=80&w=1200", "Pop dance upbeat romantic"),
        ("mv003", "Dune: Part Two", "Netflix", "Sci-Fi", "https://images.unsplash.com/photo-1502082553048-f009c37129b9?q=80&w=1200", "Desert power prophecy spice war saga"),
        ("pr003", "Apple Watch Series 9", "Amazon", "Wearables", "https://images.unsplash.com/photo-1515041219749-89347f83291a?q=80&w=1200", "Health fitness smartwatch notifications"),
        ("mu003", "Naatu Naatu", "Spotify", "Dance", "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?q=80&w=1200", "Energetic dance Indian beat"),
        ("mv004", "The Social Network", "Netflix", "Drama", "https://images.unsplash.com/photo-1448932223592-d1fc686e76ea?q=80&w=1200", "Startup friendship betrayal coding"),
        ("pr004", "Canon EOS R50", "Amazon", "Cameras", "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?q=80&w=1200", "Mirrorless camera creator vlogging"),
        ("mu004", "Arjit Mix", "Spotify", "Romance", "https://images.unsplash.com/photo-1506157786151-b8491531f063?q=80&w=1200", "Romantic mellow vocals Hindi"),
    ]
    return pd.DataFrame(data, columns=["item_id","title","provider","genre","image","text"])

def _text_embed(texts: List[str], dim: int = 64) -> np.ndarray:
    # Simple hashing-based text embedding for demo (content-based arm of hybrid)
    rng = np.random.RandomState(13)
    V = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t)) % (10**8)
        rng.seed(h)
        V[i] = rng.rand(dim)
    return V

# -------------------- Public Load --------------------

def load_item_embeddings(items: Optional[pd.DataFrame], artifacts_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, Dict[str,int], np.ndarray]:
    """
    Load catalog + embeddings. If artifacts present (items.npy, embs.npy, A.npy), load them.
    Else: build a synthetic catalog with hybrid embeddings (text + random CF).
    Returns: (items_df, embeddings, id_to_idx, adjacency A)
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    items_path = artifacts_dir / "items.csv"
    embs_path = artifacts_dir / "embs.npy"
    A_path = artifacts_dir / "A.npy"

    if items is None:
        if items_path.exists():
            items_df = pd.read_csv(items_path)
        else:
            items_df = _demo_items()
            items_df.to_csv(items_path, index=False)
    else:
        items_df = items.copy()
        items_df.to_csv(items_path, index=False)

    N = len(items_df)
    dim = 64

    if embs_path.exists() and A_path.exists():
        E = np.load(embs_path)
        A = np.load(A_path)
        if E.shape[0] != N:
            # regenerate to keep shapes consistent
            raise ValueError("embs.npy count mismatch with items.csv")
    else:
        # Build hybrid: content text embedding + CF-like noise
        T = _text_embed(items_df["text"].fillna("").tolist(), dim=dim)
        CF = np.random.RandomState(5).rand(N, dim).astype(np.float32) * 0.35
        E = (0.65 * T + 0.35 * CF).astype(np.float32)
        # Adjacency: connect items by shared provider/genre
        A = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i == j: 
                    continue
                score = 0.0
                if items_df.iloc[i]["provider"] == items_df.iloc[j]["provider"]:
                    score += 1.0
                if items_df.iloc[i]["genre"] == items_df.iloc[j]["genre"]:
                    score += 1.0
                A[i, j] = score
        # row-normalize
        row_sum = A.sum(axis=1, keepdims=True) + 1e-8
        A = A / row_sum
        np.save(embs_path, E)
        np.save(A_path, A)

    id_to_idx = {iid: i for i, iid in enumerate(items_df["item_id"].tolist())}

    # Quanta GNN propagation
    qgnn = QuantaGNN(num_items=N, dim=E.shape[1])
    E_prop = qgnn.propagate(A, E, steps=2)
    E_prop = _normalize_rows(E_prop)

    return items_df, E_prop, id_to_idx, A

# -------------------- User Vector --------------------

def make_user_vector(liked: Iterable[str], bagged: Iterable[str], id_to_idx: Dict[str,int], E: np.ndarray) -> np.ndarray:
    idxs = []
    w = []
    for iid in liked:
        if iid in id_to_idx:
            idxs.append(id_to_idx[iid])
            w.append(1.0)
    for iid in bagged:
        if iid in id_to_idx:
            idxs.append(id_to_idx[iid])
            w.append(0.7)
    if not idxs:
        # cold vector: small noise
        return np.ones(E.shape[1], dtype=np.float32) * (1.0 / E.shape[1])
    V = (E[idxs] * np.array(w)[:, None]).mean(axis=0)
    V = V / (np.linalg.norm(V) + 1e-8)
    return V

# -------------------- Recommend --------------------

def recommend_items(user_vec: np.ndarray,
                    E: np.ndarray,
                    items_df: pd.DataFrame,
                    exclude: set[str] = set(),
                    topk: int = 12,
                    A: Optional[np.ndarray] = None,
                    crowd: Optional[List[dict]] = None,
                    force_content: bool = False) -> List[str]:
    """
    Hybrid: cosine scores; optional crowd prior; optional content-only mode.
    """
    scores = _cosine_scores(user_vec, E)

    # crowd boosting (collaborative prior)
    if crowd and not force_content:
        pop = {}
        for r in crowd:
            k = r.get("item_id")
            if not k: continue
            pop[k] = pop.get(k, 0) + (2.0 if r.get("action") == "like" else 1.0)
        maxp = max(pop.values()) if pop else 1.0
        for i, iid in enumerate(items_df["item_id"]):
            if iid in pop:
                scores[i] += 0.15 * (pop[iid] / maxp)

    # Mask excluded
    mask = np.array([iid not in exclude for iid in items_df["item_id"]], dtype=bool)
    idxs = np.argsort(-scores)
    filtered = [items_df["item_id"].iloc[i] for i in idxs if mask[i]]
    return filtered[:topk]

# -------------------- Cold Start (MMR) --------------------

def cold_start_mmr(items_df: pd.DataFrame, E: np.ndarray, lambda_: float = 0.65, k: int = 12) -> List[str]:
    """
    MMR: pick diverse set w.r.t cosine similarity and content variety, no user profile needed.
    """
    N = len(items_df)
    if N <= k:
        return items_df["item_id"].tolist()

    picked = []
    cand = list(range(N))
    # reference centroid (popularity-agnostic)
    ref = E.mean(axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-8)

    sim_to_ref = (E @ ref)
    first = int(np.argmax(sim_to_ref))
    picked.append(first)
    cand.remove(first)

    while len(picked) < k and cand:
        best_i = cand[0]
        best_score = -1e9
        for i in cand:
            rel = sim_to_ref[i]
            div = 0.0 if not picked else max(E[i] @ E[j] for j in picked)
            mmr = lambda_ * rel - (1 - lambda_) * div
            if mmr > best_score:
                best_score = mmr
                best_i = i
        picked.append(best_i)
        cand.remove(best_i)

    return items_df["item_id"].iloc[picked].tolist()

# -------------------- Compare Metrics --------------------

def diversity_personalization_novelty(df: pd.DataFrame,
                                      user_vec: np.ndarray,
                                      E: np.ndarray,
                                      id2idx: Dict[str,int]) -> Tuple[float,float,float]:
    """
    Diversity: 1 - avg pairwise cosine
    Personalization: avg cosine(user_vec, items)
    Novelty: 1 - normalized item frequency over provider/genre (proxy)
    """
    if df.empty:
        return (0.0, 0.0, 0.0)

    idxs = [id2idx[i] for i in df["item_id"] if i in id2idx]
    X = E[idxs]
    # diversity
    if len(idxs) > 1:
        sims = []
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                sims.append(X[i] @ X[j])
        div = 1.0 - float(np.mean(sims))
    else:
        div = 1.0

    # personalization
    per = float(np.mean(X @ (user_vec / (np.linalg.norm(user_vec)+1e-8))))

    # novelty (proxy): penalize common provider/genre clusters
    prov_counts = df["provider"].value_counts().to_dict()
    gen_counts = df["genre"].value_counts().to_dict()
    prov_pen = np.mean([prov_counts[p] for p in df["provider"]])
    gen_pen = np.mean([gen_counts[g] for g in df["genre"]])
    max_pen = max(prov_pen, gen_pen, 1.0)
    nov = 1.0 - (max_pen / max(len(df), 1))

    # clamp
    div = float(np.clip(div, 0.0, 1.0))
    per = float(np.clip(per, 0.0, 1.0))
    nov = float(np.clip(nov, 0.0, 1.0))
    return (div, per, nov)
