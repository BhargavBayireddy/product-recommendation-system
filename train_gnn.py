# train_gnn.py
import json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import scipy.sparse as sp
from models.lightgcn import LightGCN, build_norm_adj, bpr_loss

BASE = Path(__file__).parent
ART  = BASE / "artifacts"
ART.mkdir(exist_ok=True)

EMB_DIM = 32
LAYERS  = 3
LR      = 1e-2
EPOCHS  = 4
BATCH   = 4096
SEED    = 42
random.seed(SEED); np.random.seed(SEED)

def synth_offline_dataset(n_users=600, n_items=900, avg_deg=30):
    rng = np.random.default_rng(SEED)
    # items meta (3 domains, moods)
    domains = np.array(["netflix","amazon","spotify"])
    moods   = np.array(["focus","relax","fitness","happiness","engaged","calm"])
    items = pd.DataFrame({
        "item_id": [f"it_{i}" for i in range(n_items)],
        "name":    [f"Item {i:04d}" for i in range(n_items)],
        "domain":  rng.choice(domains, size=n_items, replace=True),
        "category": rng.choice(["entertainment","product","music"], size=n_items, replace=True),
        "mood":    rng.choice(moods, size=n_items, replace=True),
        "goal":    lambda df: df["mood"]
    })
    items["goal"] = items["mood"]

    # bipartite interactions
    edges = []
    for u in range(n_users):
        k = rng.poisson(avg_deg)
        k = int(max(10, min(60, k)))
        picks = rng.choice(n_items, size=k, replace=False)
        for i in picks:
            edges.append((u, i))
    inter = pd.DataFrame({"user_id":[f"u_{u}" for u,_ in edges],
                          "item_id":[items.iloc[i]["item_id"] for _,i in edges]})
    return items, inter

def _sample_batch(ui_dict, nu, ni, bs):
    rng = np.random.default_rng(SEED)
    U,P,N = [],[],[]
    for _ in range(bs):
        u = rng.integers(0, nu)
        if not ui_dict[u]:
            continue
        p = rng.choice(ui_dict[u])
        n = rng.integers(0, ni)
        tries = 0
        while n in ui_dict[u] and tries < 10:
            n = rng.integers(0, ni); tries += 1
        U.append(u); P.append(int(p)); N.append(int(n))
    if not U:
        U=[0]; P=[0]; N=[1]
    return (torch.tensor(U, dtype=torch.long),
            torch.tensor(P, dtype=torch.long),
            torch.tensor(N, dtype=torch.long))

def main():
    print("🔧 Building small offline dataset...")
    items, inter = synth_offline_dataset(n_users=600, n_items=900, avg_deg=30)

    # index maps
    users = sorted(inter["user_id"].unique())
    items_ids = sorted(items["item_id"].unique())
    uid2idx = {u:i for i,u in enumerate(users)}
    iid2idx = {it:i for i,it in enumerate(items_ids)}
    inter["u_idx"] = inter["user_id"].map(uid2idx).astype(int)
    inter["i_idx"] = inter["item_id"].map(iid2idx).astype(int)
    ui_edges = list(zip(inter["u_idx"].tolist(), inter["i_idx"].tolist()))

    print(f"Data  : users={len(uid2idx)} items={len(iid2idx)} edges={len(ui_edges)}")

    norm_adj = build_norm_adj(len(uid2idx), len(iid2idx), ui_edges)
    ui_dict = {u:[] for u in range(len(uid2idx))}
    for u,i in ui_edges: ui_dict[u].append(i)

    model = LightGCN(len(uid2idx), len(iid2idx), EMB_DIM, norm_adj, LAYERS)
    opt = optim.Adam(model.parameters(), lr=LR)

    steps = max(1, math.ceil(len(ui_edges) / max(512, BATCH)))
    print(f"Train : epochs={EPOCHS} steps/epoch={steps}")
    model.train()
    for ep in range(1, EPOCHS+1):
        loss_sum, cnt = 0.0, 0
        for _ in range(steps):
            U,P,N = _sample_batch(ui_dict, len(uid2idx), len(iid2idx), BATCH)
            Uz, Iz = model()
            loss = bpr_loss(Uz[U], Iz[P], Iz[N])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); cnt += 1
        print(f"Epoch {ep:02d}/{EPOCHS} loss={loss_sum/max(1,cnt):.4f}")

    model.eval()
    with torch.no_grad():
        Uz, Iz = model()
    ART.mkdir(exist_ok=True)
    np.save(ART/"user_embeddings.npy", Uz.cpu().numpy())
    np.save(ART/"item_embeddings.npy", Iz.cpu().numpy())
    with open(ART/"idmaps.json","w",encoding="utf-8") as f:
        json.dump({"uid2idx": uid2idx, "iid2idx": iid2idx}, f, indent=2)
    items.to_csv(ART/"items_snapshot.csv", index=False)
    print("✅ Done. Saved to artifacts/")

if __name__ == "__main__":
    main()
