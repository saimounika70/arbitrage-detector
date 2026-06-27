import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from gnn_model import ArbitrageGNN

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "INR"]
N = len(CURRENCIES)

def bellman_ford_labels(rates):
    nodes = list(range(N))
    in_cycle = [0] * N
    edges = []
    for u in range(N):
        for v in range(N):
            if u != v and rates[u][v] > 0:
                edges.append((u, v, -math.log(rates[u][v])))

    for source in nodes:
        dist = [float('inf')] * N
        pred = [-1] * N
        dist[source] = 0.0
        for _ in range(N - 1):
            for u, v, w in edges:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u
        cycle_node = -1
        for u, v, w in edges:
            if dist[u] + w < dist[v] - 1e-9:
                cycle_node = v
                pred[v] = u
                break
        if cycle_node == -1:
            continue
        visited = cycle_node
        for _ in range(N):
            visited = pred[visited]
        node = visited
        for _ in range(N):
            in_cycle[node] = 1
            node = pred[node]
            if node == visited:
                break
    return in_cycle

def random_rates(inject_arbitrage=False):
    base = np.random.uniform(0.5, 2.0, size=(N, N))
    np.fill_diagonal(base, 1.0)
    for i in range(N):
        for j in range(N):
            if i != j:
                base[j][i] = 1.0 / base[i][j] * np.random.uniform(0.98, 1.02)
    if inject_arbitrage:
        i, j, k = random.sample(range(N), 3)
        boost = random.uniform(1.005, 1.02)
        base[i][j] *= boost ** (1/3)
        base[j][k] *= boost ** (1/3)
        base[k][i] *= boost ** (1/3)
    return base.tolist()

def rates_to_pyg(rates):
    x = []
    for u in range(N):
        feats = []
        for v in range(N):
            if u == v:
                feats.append(0.0)
            else:
                feats.append(math.log(max(rates[u][v], 1e-9)))
        x.append(feats)
    x = torch.tensor(x, dtype=torch.float)
    src, dst, ew = [], [], []
    for u in range(N):
        for v in range(N):
            if u != v and rates[u][v] > 0:
                src.append(u)
                dst.append(v)
                ew.append(-math.log(rates[u][v]))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(ew, dtype=torch.float)
    labels = bellman_ford_labels(rates)
    y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

def train():
    print("Generating synthetic training data...")
    dataset = []
    for i in range(3000):
        inject = (i % 3 == 0)
        rates = random_rates(inject_arbitrage=inject)
        data = rates_to_pyg(rates)
        dataset.append(data)
    print(f"Generated {len(dataset)} graphs")

    total_nodes = sum(d.y.sum().item() for d in dataset)
    print(f"Nodes in arbitrage cycles: {int(total_nodes)} / {len(dataset)*N}")

    model = ArbitrageGNN(in_channels=N, hidden_channels=32, out_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0
        for data in dataset:
            optimizer.zero_grad()
            raw = model(data.x, data.edge_index, data.edge_attr)
            out = torch.clamp(raw, min=1e-7, max=1.0 - 1e-7)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {total_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), "gnn_weights.pt")
    print("Model saved to gnn_weights.pt")

if __name__ == "__main__":
    train()
