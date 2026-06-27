# 💱 AI-Accelerated Currency Arbitrage Detector


<img width="1512" height="945" alt="Screenshot 2026-06-27 at 5 26 26 AM" src="https://github.com/user-attachments/assets/5cbd0ef2-b510-4a3c-8952-d903390e640a" />


A real-time currency arbitrage detection system that uses a **Graph Neural Network (GNN)** heuristic to intelligently prune the search space of the O(VE) Bellman-Ford algorithm, finding profitable FX trading cycles at a fraction of the computational cost.

## 🚀 GNN Algorithmic Speedup

Traditional arbitrage detection treats currency exchange rates as a complete, weighted directed graph, running the Bellman-Ford algorithm from every single node to find negative cycles (which represent risk-free profit). This has a time complexity of `O(V * VE)`, which scales poorly as the number of monitored currencies grows.

**This project solves that scalability issue using Deep Learning.** 
A custom Graph Neural Network (implemented in PyTorch Geometric) is trained to evaluate the graph topology and predict which currency nodes are statistically likely to be part of a negative cycle. 

By using the GNN as a pre-filter, the system safely ignores irrelevant nodes and only executes Bellman-Ford from the flagged subset, delivering a **massive algorithmic speedup** in real-time.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, PyTorch, PyTorch Geometric
- **Frontend:** React, TypeScript, Recharts
- **External API:** ExchangeRate-API (Live FX Data)

## ⚙️ How it Works

1. **Live Data Ingestion:** Fetches real-time FX rates.
2. **Graph Construction:** Currencies are modeled as nodes. Exchange rates become directed edges with a weight of `-log(rate)`.
3. **AI Heuristic Pruning:** The PyTorch GNN evaluates the graph in a single forward pass, predicting the nodes most likely to be part of an arbitrage cycle.
4. **Deep Search:** The Bellman-Ford algorithm is selectively executed only on the GNN-flagged nodes.
5. **Cycle Extraction & Verification:** The identified negative cycle is traced and the real-world profit percentage is calculated.

## 💻 Local Setup

### Prerequisites
- Node.js & npm
- Python 3.9+
- An API key from [ExchangeRate-API](https://www.exchangerate-api.com/)

### Backend Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your API key:
   Ensure you have a `.env` file in the `backend` directory containing your key:
   ```env
   API_KEY=your_exchangerate_api_key_here
   ```
5. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```

*(The frontend will automatically open at `http://localhost:3000`)*

## 🧠 Retraining the GNN

If you wish to retrain the Graph Neural Network on newly generated synthetic graphs:
```bash
cd backend
source venv/bin/activate
python train_gnn.py
```
This will run the training loop for 50 epochs and save the updated weights to `gnn_weights.pt`.
