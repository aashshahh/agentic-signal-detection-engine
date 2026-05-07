<div align="center">

<!-- ANIMATED TITLE -->
<img src="https://readme-typing-svg.demolab.com?font=Space+Mono&size=28&duration=3000&pause=1000&color=3B82F6&center=true&vCenter=true&width=700&lines=Signal+Detection+Engine;Detect+market+moves+before+they+happen;7-Node+Agentic+AI+Pipeline;XGBoost+%2B+SBERT+%2B+Bayesian" alt="Typing SVG" />

<br/>

**A real-time multi-agent AI system that detects early narrative shifts in Reddit and HackerNews _before_ prediction market prices move.**

<br/>

<!-- BADGES ROW 1 -->
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-ec6c00?style=flat-square)](https://xgboost.ai)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.62-1a7f64?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

<!-- BADGES ROW 2 -->
[![WebSocket](https://img.shields.io/badge/WebSocket-Live%20Streaming-00e5a0?style=flat-square)](https://fastapi.tiangolo.com/advanced/websockets/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat-square)](https://trychroma.com)
[![SBERT](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-blueviolet?style=flat-square)](https://sbert.net)
[![Deploy](https://img.shields.io/badge/Deploy-Render%20%7C%20Railway-brightgreen?style=flat-square)](render.yaml)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions&logoColor=white)](.github/workflows/ci.yml)

<br/>

<!-- LINKS -->
[**🌐 Live Demo**](https://aashjatin.github.io/agentic-signal-detection-engine) · [**📊 Dashboard**](https://your-app.onrender.com) · [**📄 API Docs**](https://your-app.onrender.com/docs) · [**📑 Research Report**](AashJatinShah_report.pdf)

<br/>

</div>

---

## 🧠 The Core Idea

> **Before a prediction market price moves, the narrative is already shifting.**

Information rarely arrives in complete, actionable form. It diffuses gradually — through Reddit threads, HackerNews discussions, and subtle changes in how events are described. Most systems look at price history. **We look 14 minutes earlier.**

```
Reddit + HackerNews + Polymarket
         ↓
  4 Signal Features (SBERT + NLP)
         ↓
  XGBoost Classifier (0.81 AUC)
         ↓
  Bayesian Posterior P(market shift)
         ↓
  🚨 Alert fires → Dashboard updates live via WebSocket
```

---

## 📋 Table of Contents

- [Core Idea](#-the-core-idea)
- [Features](#-features)
- [Architecture](#-architecture)
- [4 Signal Feature Families](#-4-signal-feature-families)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Dashboard](#-dashboard)
- [API Reference](#-api-reference)
- [Deploy to Cloud](#-deploy-to-cloud)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Academic Context](#-academic-context)
- [Authors](#-authors)

---

## ✨ Features

| Feature | Description |
|---|---|
| **🔴 Live Polymarket Data** | Pulls real market probabilities, volume, liquidity via public API — no key needed |
| **📡 WebSocket Streaming** | Dashboard updates the instant the pipeline runs — not polling, true push |
| **🤖 7-Node LangGraph Pipeline** | Modular agentic architecture — each node is independently testable |
| **📐 SBERT Embedding Drift** | Detects semantic directional shift in social media discussion |
| **🌀 Topic Entropy** | Shannon entropy over n-gram distribution — low entropy = converging narrative |
| **🔍 Rare Phrase Emergence** | Catches new terminology entering the discourse before it goes mainstream |
| **⚡ Engagement Burst Z-Score** | Statistical anomaly detection on post volume — sometimes it's *how fast*, not *what* |
| **📊 XGBoost + SHAP** | Calibrated probabilistic scores + SHAP feature attribution for every prediction |
| **🎲 Bayesian Calibration** | Beta-Binomial posterior informed by Le (2026) calibration decomposition |
| **🚀 One-Command Deploy** | Render/Railway deploy in under 5 minutes via `render.yaml` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1 — DATA INGESTION                     │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  Polymarket API  │  │  Reddit (8 subs) │  │  HackerNews  │  │
│  │  live markets    │  │  r/wsb, r/invest │  │  Algolia API │  │
│  │  probabilities   │  │  r/stocks + more │  │  keyword     │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘  │
└───────────┼────────────────────┼───────────────────┼───────────┘
            └────────────────────┴───────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2 — PROCESSING                         │
│                                                                 │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │   Embedding Agent   │      │      Feature Agent          │   │
│  │   SBERT → ChromaDB  │  →   │  δ drift · H entropy       │   │
│  │   384-dim vectors   │      │  ρ emergence · z burst      │   │
│  └─────────────────────┘      └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3 — INTELLIGENCE                       │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  XGBoost         │  │  Bayesian Agent  │  │  LLM Reasoner│  │
│  │  P(signal) score │  │  Beta-Binomial   │  │  Rule-based  │  │
│  │  SHAP attribution│  │  Le 2026 priors  │  │  + Ollama    │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 4 — OUTPUT                             │
│  WebSocket Broadcast · JSON Report · 🚨 Alert (P > 70%)         │
│  7-Page Real-Time Dashboard · REST API · CSV Export             │
└─────────────────────────────────────────────────────────────────┘
```

**Orchestrated end-to-end with [LangGraph](https://langchain-ai.github.io/langgraph/) — auto-runs every 5 minutes.**

---

## 📐 4 Signal Feature Families

These four features compose the XGBoost input vector `x = [δ, H, ρ, z]`:

### Feature 1 · Embedding Drift `δ`
Split posts at temporal midpoint. Encode each with Sentence-BERT. Measure cosine distance between period means:
```
δ_k = 1 − cos(v̄_early, v̄_late)
```
> δ > 0.40 → significant semantic directional shift · **Highest SHAP weight: 0.26**

### Feature 2 · Topic Entropy `H`
Shannon entropy over top-30 n-gram frequency distribution:
```
H_k = −Σ p_i · log(p_i)
```
> Low H → converging, coordinated narrative → pre-announcement pattern · **SHAP: 0.18**

### Feature 3 · Rare Phrase Emergence `ρ`
N-grams with historical frequency < 3 that surge in the recent window:
```
ρ_k = Σ c_new(g) / (c_old(g) + 1)  for g where c_old < 3
```
> Catches new terminology entering the discourse — characteristic of information leaking in · **SHAP: 0.22**

### Feature 4 · Engagement Burst Z-Score `z`
Rolling z-score over hourly aggregate post engagement:
```
z_k = (x̄_recent − μ_roll) / σ_roll
```
> z > 2.0 → statistically unusual coordination spike · **SHAP: 0.12**

---

## 📈 Results

| Metric | Value | Notes |
|---|---|---|
| **XGBoost ROC-AUC** | **0.812** | 5-fold stratified CV |
| LR Baseline AUC | 0.673 | Logistic Regression |
| Random Baseline | 0.500 | Chance classifier |
| **AUC Lift** | **+0.139** | Over LR baseline |
| **Avg Lead Time** | **+14.3 min** | Signal → market price move |
| Lead Time Std Dev | 5.8 min | Stability across runs |
| Bayesian Alert Threshold | > 70% | Posterior mean P(shift) |
| Pipeline Cadence | 5 min | Auto background scheduler |

> **Consistently positive lead times** validate the core hypothesis: measurable narrative changes in online discourse precede prediction market price adjustments.

<details>
<summary><strong>SHAP Feature Importance</strong></summary>

```
Embedding Drift     ████████████████████████████   0.26  ← strongest
Rare Phrase Emerg.  ████████████████████████        0.22
Topic Entropy       ██████████████████              0.18
Engagement Burst    ████████████                    0.12
Baseline            ████████████████████████████   0.25
```

</details>

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- No API keys required for core functionality

### 1. Clone & Install

```bash
git clone https://github.com/AashJatin/agentic-signal-detection-engine.git
cd agentic-signal-detection-engine
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
```

```env
# All optional — system works without any keys (demo mode)
REDDIT_CLIENT_ID=your_id          # reddit.com/prefs/apps
REDDIT_CLIENT_SECRET=your_secret
NEWSAPI_KEY=your_key               # newsapi.org (free tier)
OLLAMA_API_KEY=your_key            # ollama.ai (for LLM reasoning)
```

### 3. Run

```bash
python run.py
```

Open **[http://localhost:8000](http://localhost:8000)** 🎉

> **No keys needed.** Polymarket, HackerNews, and Reddit are all public APIs. The system starts in demo mode with realistic mock data and upgrades to live data automatically when APIs respond.

---

## 🖥️ Dashboard

The dashboard has **7 analytical pages**, all updated live via WebSocket:

| Page | What it shows |
|---|---|
| **Macro Dashboard** | Live market probabilities, signal badges, drift bars, Bayesian ring, pipeline log |
| **Signal Feed** | Per-keyword breakdown: all 4 features, SHAP values, top posts |
| **Bayesian Model** | Beta distribution chart, prior→posterior evolution, 95% CI |
| **SHAP Features** | Global importance bar chart, per-feature descriptions |
| **Backtest** | XGBoost vs LR vs Random table, lead-time histogram |
| **Drift Monitor** | Full drift bar chart with interpretation guide |
| **Data Sources** | Live HN posts, Reddit feed, full Polymarket market table |

> Built with vanilla JS + Chart.js — no framework. Single HTML file. Deploys anywhere.

---

## 🔌 API Reference

```
GET  /api/markets         Live Polymarket markets
GET  /api/signals         Signal scores, Bayesian state, SHAP values
GET  /api/search?q=...    Live keyword search + narrative posts
GET  /api/backtest        Model comparison + lead-time stats
GET  /api/stats           System state: runs, posterior, SBERT status
POST /api/run             Trigger manual pipeline execution
WS   /ws                  Real-time stream: markets, signals, logs
GET  /docs                Interactive Swagger UI
```

<details>
<summary><strong>Example: GET /api/signals</strong></summary>

```json
{
  "signals": [
    {
      "keyword": "federal reserve",
      "prob": 0.812,
      "label": "SIGNAL",
      "drift": 0.64,
      "entropy": 0.31,
      "emergence": 1.8,
      "burst_z": 2.3,
      "shap": {
        "drift_score": 0.26,
        "topic_entropy": 0.18,
        "emergence_score": 0.22,
        "burst_z": 0.12
      }
    }
  ],
  "bayesian_prob": 0.74,
  "alert": true
}
```

</details>

---

## ☁️ Deploy to Cloud

### Render (recommended — free tier, 5 min)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

Or manually:
```bash
# 1. Push to GitHub
# 2. render.com → New Web Service → connect repo
# 3. Build: pip install -r requirements.txt
# 4. Start: uvicorn backend.server:app --host 0.0.0.0 --port $PORT
```

The `render.yaml` in this repo enables one-click deploy.

### Railway

```bash
npm install -g @railway/cli
railway login && railway new && railway up
```

### Docker

```bash
docker build -t signal-engine .
docker run -p 8000:8000 signal-engine
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Orchestration** | LangGraph 0.0.62 | 7-node stateful agent graph |
| **Backend** | FastAPI + uvicorn | REST API + WebSocket streaming |
| **Markets** | Polymarket Gamma API | Live prediction market data |
| **Narrative** | Reddit JSON + HN Algolia | Social media narrative ingestion |
| **Embeddings** | Sentence-BERT (all-MiniLM-L6-v2) | 384-dim semantic vectors |
| **Vector Store** | ChromaDB | Persisted embedding storage + RAG |
| **ML Model** | XGBoost 2.0 + SMOTE | Binary signal classification |
| **Explainability** | SHAP | Feature attribution per prediction |
| **Calibration** | Beta-Binomial Bayesian | Self-updating P(market shift) |
| **Frontend** | Vanilla JS + Chart.js | Real-time dashboard (no framework) |
| **Deploy** | Render / Railway / Docker | Cloud deployment |

---

## 📁 Project Structure

```
agentic-signal-detection-engine/
│
├── agents/
│   ├── ingestion/
│   │   ├── polymarket_agent.py   ← Live markets + volatility labeling
│   │   ├── reddit_agent.py       ← PRAW scraper (8 subreddits)
│   │   └── news_agent.py         ← NewsAPI + RSS feeds
│   ├── processing/
│   │   ├── embedding_agent.py    ← SBERT → ChromaDB + drift scoring
│   │   └── feature_agent.py      ← All 4 signal feature families
│   └── intelligence/
│       ├── prediction_agent.py   ← XGBoost + SMOTE + SHAP
│       └── bayesian_llm_agent.py ← Beta-Binomial + LLM reasoning
│
├── backend/
│   └── server.py                 ← FastAPI + WebSocket server
│
├── pipeline/
│   └── graph.py                  ← LangGraph 7-node orchestration
│
├── frontend/
│   └── public/index.html         ← Full 7-page dashboard (single file)
│
├── website/                      ← GitHub Pages demo site
│   └── index.html
│
├── notebooks/
│   └── exploration.ipynb         ← EDA + ROC + Bayesian posterior plots
│
├── docs/
│   └── EMT678A_Report_AashJatinShah.pdf
│
├── data/                         ← ChromaDB + model artifacts (gitignored)
├── config.py                     ← Central configuration
├── main.py                       ← CLI entry point
├── run.py                        ← Quick start script
├── requirements.txt
├── render.yaml                   ← One-click Render deploy
├── Dockerfile
└── .env.example
```

---

## 📚 Academic Context

This project was built for **EMT 678A — Big Data Technologies** at Stevens Institute of Technology. The system design is grounded in recent empirical work:

> Le (2026) analyzed 292 million trades across 327,000 contracts on Kalshi and Polymarket, showing that a 70-cent contract in a political prediction market one week before resolution corresponds to a **true probability of ~83%**, not 70%. This 13-point gap defines the exploitable window our system targets.

The Bayesian prior design is directly informed by this calibration decomposition — political market keywords receive `Beta(3,2)` priors rather than the standard `Beta(2,2)`.

**References:**
- Le, N.A. (2026). *Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets.* arXiv:2602.19520
- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP

---

## 👤 Authors

<table>
<tr>
<td align="center">
<strong>Aash Jatin Shah</strong><br/>
<sub>XGBoost · SHAP · Reddit · FastAPI<br/>WebSocket Dashboard · Bayesian Calibration<br/>Deployment Infrastructure</sub><br/>
<a href="mailto:ashah16@stevens.edu">ashah16@stevens.edu</a>
</td>
<td align="center">
<strong>Bhavesh Bholanath Maurya</strong><br/>
<sub>PELT Changepoint · LangGraph<br/>RAG Pipeline · LLM Reasoning<br/>HackerNews Ingestion · Streamlit</sub><br/>
<a href="https://github.com/BhaveshMRA/agentic-signal-detection-engine">BhaveshMRA/agentic-signal-detection-engine</a>
</td>
</tr>
</table>

---

<div align="center">

**Stevens Institute of Technology · EMT 678A Big Data Technologies · Spring 2026**

*Not financial advice. For academic purposes only.*

[![Star History Chart](https://api.star-history.com/svg?repos=AashJatin/agentic-signal-detection-engine&type=Date)](https://star-history.com/#AashJatin/agentic-signal-detection-engine&Date)

</div>
