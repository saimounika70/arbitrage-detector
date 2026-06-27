"""
GNN inference for arbitrage detection.

The GNN predicts which currency nodes are likely in an arbitrage cycle.
We then run Bellman-Ford only from those high-scoring nodes — not all 8.
This is the speed advantage: O(k * VE) instead of O(V * VE) where k << V.
"""

import math
import torch
from torch_geometric.data import Data
from gnn_model import ArbitrageGNN

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "INR"]
N = len(CURRENCIES)

# load model once at module level (don't reload on every request)
_model = None

def load_model(weights_path="gnn_weights.pt"):
    global _model
    if _model is None:
        _model = ArbitrageGNN(in_channels=N, hidden_channels=32, out_channels=1)
        _model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        _model.eval()
    return _model


def rates_to_pyg(rates):
    """Same conversion as in training."""
    x = []
    for u in CURRENCIES:
        feats = []
        for v in CURRENCIES:
            if u == v:
                feats.append(0.0)
            else:
                feats.append(math.log(max(rates[u][v], 1e-9)))
        x.append(feats)
    x = torch.tensor(x, dtype=torch.float)

    src, dst, ew = [], [], []
    for i, u in enumerate(CURRENCIES):
        for j, v in enumerate(CURRENCIES):
            if u != v and rates[u][v] > 0:
                src.append(i)
                dst.append(j)
                ew.append(-math.log(rates[u][v]))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(ew, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


def predict_arbitrage_nodes(rates, threshold=0.5):
    """
    Run GNN on the rate graph.
    Returns list of currency names the GNN thinks are in arbitrage cycles.
    Also returns raw scores for all nodes (useful for visualisation).
    """
    model = load_model()
    data = rates_to_pyg(rates)

    with torch.no_grad():
        scores = model(data.x, data.edge_index, data.edge_attr)

    scores = scores.squeeze().tolist()
    node_scores = {CURRENCIES[i]: round(scores[i], 4) for i in range(N)}

    # only run Bellman-Ford from nodes the GNN flagged
    flagged = [CURRENCIES[i] for i in range(N) if scores[i] >= threshold]

    return flagged, node_scores
