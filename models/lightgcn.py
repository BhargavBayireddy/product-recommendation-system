# models/lightgcn.py
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

def build_norm_adj(num_users, num_items, ui_edges):
    n_nodes = num_users + num_items
    rows, cols, data = [], [], []

    for u, i in ui_edges:
        vi = num_users + i
        rows.append(u);  cols.append(vi); data.append(1.0)
        rows.append(vi); cols.append(u);  data.append(1.0)

    mat = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    deg = np.array(mat.sum(1)) + 1e-7
    inv_sqrt = np.power(deg, -0.5).flatten()
    D_inv = sp.diags(inv_sqrt)

    norm_adj = D_inv @ mat @ D_inv
    return norm_adj.tocsr()

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, norm_adj, n_layers, device="cpu"):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim   = emb_dim
        self.n_layers  = n_layers

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.register_buffer("norm_adj", self._to_torch_sparse(norm_adj).to(device))

    def _to_torch_sparse(self, mat):
        coo = mat.tocoo()
        idx = torch.tensor([coo.row, coo.col], dtype=torch.long)
        data = torch.tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, data, coo.shape)

    def forward(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        out = torch.stack(embs, dim=0).mean(dim=0)
        users = out[:self.num_users]
        items = out[self.num_users:]
        return users, items

def bpr_loss(u, p, n):
    pos = (u * p).sum(dim=1)
    neg = (u * n).sum(dim=1)
    return -torch.mean(torch.log(torch.sigmoid(pos - neg) + 1e-7))
