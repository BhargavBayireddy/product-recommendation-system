# prep_fast_embeddings.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
ART  = BASE / "artifacts"
ART.mkdir(exist_ok=True)

ITEMS_CSV = ART / "items_snapshot.csv"
ITEM_EMB  = ART / "item_embeddings.npy"
IDMAPS    = ART / "idmaps.json"

def main():
    if not ITEMS_CSV.exists():
        raise SystemExit("Missing artifacts/items_snapshot.csv. Run the app once or data_real.py to build it.")

    items = pd.read_csv(ITEMS_CSV)
    items = items.dropna(subset=["item_id"]).reset_index(drop=True)
    items["item_id"] = items["item_id"].astype(str)
    ids = items["item_id"].tolist()

    rng = np.random.default_rng(777)
    embs = rng.normal(0, 0.1, size=(len(ids), 32)).astype(np.float32)

    iid2idx = {it: i for i, it in enumerate(ids)}
    with open(IDMAPS, "w", encoding="utf-8") as f:
        json.dump({"iid2idx": iid2idx}, f, indent=2)
    np.save(ITEM_EMB, embs)

    print(f"Saved {len(ids)} embeddings to {ITEM_EMB}")
    print(f"Saved id map to {IDMAPS}")

if __name__ == "__main__":
    main()
