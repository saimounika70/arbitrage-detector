import math
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "INR"]

def get_rates():
    """Fetch live exchange rates and build the graph."""
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/USD"
    response = requests.get(url, timeout=5)
    data = response.json()
    rates_from_usd = data["conversion_rates"]

    # build rate[i][j] = how much of currency j you get for 1 unit of currency i
    rates = {}
    for base in CURRENCIES:
        rates[base] = {}
        for target in CURRENCIES:
            if base == target:
                rates[base][target] = 1.0
            elif base == "USD":
                rates[base][target] = rates_from_usd.get(target, 0)
            else:
                # convert: base → USD → target
                base_to_usd = 1.0 / rates_from_usd.get(base, 1)
                usd_to_target = rates_from_usd.get(target, 0)
                rates[base][target] = base_to_usd * usd_to_target

    return rates


def build_graph(rates):
    """
    Build weighted graph where edge weight = -log(rate).
    A negative cycle in this graph = arbitrage opportunity.
    
    Why -log? Because:
    profit = rate1 * rate2 * rate3 > 1
    log(rate1) + log(rate2) + log(rate3) > 0
    -log(rate1) + -log(rate2) + -log(rate3) < 0
    → negative cycle!
    """
    graph = {}
    for u in CURRENCIES:
        graph[u] = {}
        for v in CURRENCIES:
            if u != v and rates[u][v] > 0:
                graph[u][v] = -math.log(rates[u][v])
    return graph


def bellman_ford(graph, source):
    """
    Standard Bellman-Ford.
    Returns (distances, predecessors, negative_cycle_node)
    negative_cycle_node is None if no arbitrage exists.
    """
    nodes = list(graph.keys())
    dist = {n: float('inf') for n in nodes}
    pred = {n: None for n in nodes}
    dist[source] = 0.0

    # relax all edges V-1 times
    for _ in range(len(nodes) - 1):
        for u in nodes:
            for v, w in graph[u].items():
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u

    # Vth relaxation — if anything still relaxes, negative cycle exists
    cycle_node = None
    for u in nodes:
        for v, w in graph[u].items():
            if dist[u] + w < dist[v] - 1e-9:
                cycle_node = v  # this node is in or reachable from a cycle
                pred[v] = u
                break
        if cycle_node:
            break

    return dist, pred, cycle_node


def extract_cycle(pred, cycle_node):
    """
    Trace back through predecessors to extract the actual cycle.
    We need to find the cycle itself, not the path to it.
    """
    # walk back 'n' steps to ensure we're inside the cycle
    visited = cycle_node
    for _ in range(len(pred)):
        visited = pred[visited]

    # now trace the cycle
    cycle = []
    node = visited
    while True:
        cycle.append(node)
        node = pred[node]
        if node == visited:
            break
    cycle.append(visited)
    cycle.reverse()
    return cycle


def compute_profit(cycle, rates):
    """Compute actual profit percentage for the cycle."""
    profit = 1.0
    for i in range(len(cycle) - 1):
        profit *= rates[cycle[i]][cycle[i+1]]
    return round((profit - 1) * 100, 4)  # as percentage


def detect_arbitrage():
    """Main function — returns detected cycles with profit."""
    rates = get_rates()
    graph = build_graph(rates)

    results = []
    seen_cycles = set()

    gnn_used = False
    gnn_nodes = []
    sources_to_check = CURRENCIES
    
    try:
        import os
        from gnn_predict import predict_arbitrage_nodes
        if os.path.exists("gnn_weights.pt"):
            flagged, node_scores = predict_arbitrage_nodes(rates, threshold=0.5)
            # If the GNN flags nodes, we only search from those
            if flagged:
                sources_to_check = flagged
            gnn_used = True
            gnn_nodes = flagged
    except Exception as e:
        print(f"GNN skipped: {e}")

    for source in sources_to_check:
        dist, pred, cycle_node = bellman_ford(graph, source)

        if cycle_node is None:
            continue

        cycle = extract_cycle(pred, cycle_node)
        if len(cycle) < 3:
            continue

        # normalise cycle for dedup
        cycle_key = tuple(sorted(cycle))
        if cycle_key in seen_cycles:
            continue
        seen_cycles.add(cycle_key)

        profit = compute_profit(cycle, rates)
        if profit > 0:
            results.append({
                "cycle": cycle,
                "profit_pct": profit,
                "path": " → ".join(cycle)
            })

    return {
        "cycles_found": len(results),
        "arbitrage": results,
        "gnn_used": gnn_used,
        "gnn_nodes_evaluated": gnn_nodes,
        "total_nodes": len(CURRENCIES),
        "rates_snapshot": {
            c: {t: round(rates[c][t], 6)
                for t in CURRENCIES if t != c}
            for c in CURRENCIES
        }
    }