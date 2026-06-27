import { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from "recharts";

const API = "https://arbitrage-detector-ll76.onrender.com";

interface Cycle {
  cycle: string[];
  profit_pct: number;
  path: string;
}

interface ArbitrageResult {
  cycles_found: number;
  arbitrage: Cycle[];
  gnn_used?: boolean;
  gnn_nodes_evaluated?: string[];
  total_nodes?: number;
}

function App() {
  const [data, setData] = useState<ArbitrageResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  const fetchArbitrage = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await axios.get(`${API}/arbitrage`);
      setData(res.data);
      setLastUpdated(new Date().toLocaleTimeString());
    } catch {
      setError("Failed to fetch — is the backend running?");
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchArbitrage();
    // auto-refresh every 30 seconds
    const interval = setInterval(fetchArbitrage, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0d1117",
      color: "#e6edf3",
      fontFamily: "monospace",
      padding: "2rem"
    }}>
      {/* Header */}
      <div style={{ marginBottom: "2rem" }}>
        <h1 style={{ fontSize: "1.8rem", fontWeight: 700, marginBottom: "0.5rem" }}>
          💱 Currency Arbitrage Detector
        </h1>
        <p style={{ color: "#8b949e", fontSize: "0.9rem" }}>
          Bellman-Ford negative cycle detection on live FX rates.
          Finds profitable currency trade cycles in real time.
        </p>
        {lastUpdated && (
          <p style={{ color: "#3fb950", fontSize: "0.8rem", marginTop: "0.5rem" }}>
            Last updated: {lastUpdated}
          </p>
        )}
      </div>

      {/* Refresh button */}
      <button
        onClick={fetchArbitrage}
        disabled={loading}
        style={{
          background: loading ? "#21262d" : "#238636",
          color: "#ffffff",
          border: "none",
          padding: "0.6rem 1.4rem",
          borderRadius: "6px",
          cursor: loading ? "not-allowed" : "pointer",
          fontSize: "0.9rem",
          marginBottom: "2rem"
        }}
      >
        {loading ? "Scanning..." : "Scan for Arbitrage"}
      </button>

      {error && (
        <div style={{
          background: "#3d1f1f", color: "#f85149",
          padding: "1rem", borderRadius: "8px", marginBottom: "1rem"
        }}>
          {error}
        </div>
      )}

      {data && (
        <>
          {/* Summary */}
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "1rem", marginBottom: "2rem"
          }}>
            <StatCard
              label="Cycles Found"
              value={data.cycles_found.toString()}
              color={data.cycles_found > 0 ? "#3fb950" : "#8b949e"}
            />
            <StatCard
              label="Currencies Monitored"
              value="8"
              color="#58a6ff"
            />
            <StatCard
              label="Best Profit"
              value={data.arbitrage.length > 0
                ? `+${Math.max(...data.arbitrage.map(c => c.profit_pct)).toFixed(4)}%`
                : "None"}
              color="#e3b341"
            />
            {data.gnn_used && data.total_nodes && data.gnn_nodes_evaluated && (
              <StatCard
                label="GNN Node Pruning"
                value={`${data.gnn_nodes_evaluated.length} / ${data.total_nodes}`}
                color="#a371f7"
              />
            )}
          </div>

          {/* Cycles */}
          {data.cycles_found === 0 ? (
            <div style={{
              background: "#161b22", border: "1px solid #30363d",
              borderRadius: "10px", padding: "2rem", textAlign: "center",
              color: "#8b949e"
            }}>
              No arbitrage cycles detected in current rates.
              Markets are efficient right now.
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              {data.arbitrage.map((cycle, i) => (
                <CycleCard key={i} cycle={cycle} rank={i + 1} />
              ))}
            </div>
          )}

          {/* How it works */}
          <div style={{
            marginTop: "2rem", background: "#161b22",
            border: "1px solid #30363d", borderRadius: "10px", padding: "1.5rem"
          }}>
            <h3 style={{ marginBottom: "1rem", color: "#58a6ff" }}>
              How it works
            </h3>
            <p style={{ color: "#8b949e", lineHeight: 1.7, fontSize: "0.9rem" }}>
              Each currency is a node. Exchange rates become edge weights
              via <code style={{ color: "#e3b341" }}>weight = -log(rate)</code>.
              A negative cycle in this graph means the product of exchange rates
              along the cycle exceeds 1 — a profitable trade loop.
              <br /><br />
              <strong style={{ color: "#e6edf3" }}>Bellman-Ford</strong> detects
              negative cycles in O(VE) time. If after V-1 relaxations we can
              still relax an edge, a negative cycle exists.
              <br /><br />
              <strong style={{ color: "#a371f7" }}>✨ GNN Heuristic Active:</strong> 
              A Graph Neural Network predicts which nodes are likely in an arbitrage cycle, 
              reducing the search space for Bellman-Ford and providing a massive speedup!
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value, color }: {
  label: string; value: string; color: string
}) {
  return (
    <div style={{
      background: "#161b22", border: "1px solid #30363d",
      borderRadius: "10px", padding: "1.2rem"
    }}>
      <div style={{ fontSize: "0.75rem", color: "#8b949e", marginBottom: "0.4rem" }}>
        {label}
      </div>
      <div style={{ fontSize: "1.5rem", fontWeight: 700, color }}>
        {value}
      </div>
    </div>
  );
}

function CycleCard({ cycle, rank }: { cycle: Cycle; rank: number }) {
  return (
    <div style={{
      background: "#161b22",
      border: `1px solid ${cycle.profit_pct > 0.1 ? "#3fb950" : "#30363d"}`,
      borderRadius: "10px", padding: "1.2rem"
    }}>
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: "0.8rem"
      }}>
        <span style={{ color: "#8b949e", fontSize: "0.8rem" }}>
          Cycle #{rank}
        </span>
        <span style={{
          background: "#1f3d2a", color: "#3fb950",
          padding: "0.2rem 0.8rem", borderRadius: "20px", fontSize: "0.85rem"
        }}>
          +{cycle.profit_pct}% profit
        </span>
      </div>
      <div style={{
        fontSize: "1rem", fontWeight: 500,
        color: "#e6edf3", letterSpacing: "0.05em"
      }}>
        {cycle.cycle.map((c, i) => (
          <span key={i}>
            <span style={{ color: "#58a6ff" }}>{c}</span>
            {i < cycle.cycle.length - 1 && (
              <span style={{ color: "#3fb950", margin: "0 0.4rem" }}>→</span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

export default App;