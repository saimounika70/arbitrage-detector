from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from arbitrage import detect_arbitrage, get_rates, CURRENCIES
import math

app = FastAPI(title="Arbitrage Detector API")

# allow React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "running", "endpoints": ["/arbitrage", "/rates", "/graph"]}

@app.get("/arbitrage")
def get_arbitrage():
    """Detect arbitrage cycles in live FX rates."""
    try:
        result = detect_arbitrage()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rates")
def get_current_rates():
    """Get current exchange rate matrix."""
    try:
        rates = get_rates()
        return {"currencies": CURRENCIES, "rates": rates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph")
def get_graph_data():
    """Return graph in format ready for frontend visualisation."""
    try:
        rates = get_rates()
        nodes = [{"id": c, "label": c} for c in CURRENCIES]
        edges = []
        for u in CURRENCIES:
            for v in CURRENCIES:
                if u != v and rates[u][v] > 0:
                    edges.append({
                        "source": u,
                        "target": v,
                        "weight": round(rates[u][v], 4),
                        "log_weight": round(-math.log(rates[u][v]), 6)
                    })
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))