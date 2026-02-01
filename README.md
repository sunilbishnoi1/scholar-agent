# 📚 Scholar Agent

> AI-powered research assistant that automates literature reviews using a multi-agent architecture

[![CI/CD](https://github.com/sunilbishnoi1/scholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sunilbishnoi1/scholar-agent/actions)
[![codecov](https://codecov.io/gh/sunilbishnoi1/scholar-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/sunilbishnoi1/scholar-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Features

- **🤖 Multi-Agent System** — LangGraph-orchestrated pipeline with 5 specialized agents (Planner → Retriever → Analyzer → Quality Checker → Synthesizer) and automatic refinement loops
- **🔍 Hybrid RAG Search** — Dense vector embeddings + BM25 keyword search with Qdrant, plus cross-encoder reranking for high-precision retrieval
- **💰 Smart Model Routing** — Cost-aware routing between Gemini models (flash-lite/flash/pro) based on task complexity and budget constraints
- **📚 Multi-Source Retrieval** — Automated paper discovery from arXiv and Semantic Scholar with intelligent deduplication
- **⚡ Real-time Updates** — WebSocket streaming with Redis Pub/Sub for live agent progress and status updates
- **🧠 Intelligent Caching** — Tiered Redis caching for LLM responses, embeddings, and search results to minimize API costs
- **📊 Observability** — OpenTelemetry tracing, structured logging, and detailed usage/cost tracking per user
- **🔐 Authentication** — JWT-based auth with user tiers, monthly budgets, and usage quotas

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                                   FRONTEND                                          │
│                        React + TypeScript + Tailwind + Zustand                      │
│                              WebSocket Client (Real-time)                           │
└──────────────────────────────────────┬─────────────────────────────────────────────┘
                                       │ REST API / WebSocket
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                 BACKEND API (FastAPI)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐   │
│  │ Auth (JWT)  │  │  Projects   │  │    Usage     │  │  WebSocket Manager        │   │
│  │  /api/auth  │  │ /api/projects│ │   Tracking   │  │  (Redis Pub/Sub)          │   │
│  └─────────────┘  └─────────────┘  └──────────────┘  └───────────────────────────┘   │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────────────┐
│                           ORCHESTRATOR (LangGraph)                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────┐  │
│  │                         State Machine & Pipeline Control                        │  │
│  │    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌─────────┐    ┌────────┐  │  │
│  │    │ Planner  │───▶│ Retriever │───▶│ Analyzer │───▶│ Quality │───▶│Synth-  │  │  │
│  │    │  Agent   │    │   Agent   │    │  Agent   │    │ Checker │    │esizer  │  │  │
│  │    └──────────┘    └───────────┘    └──────────┘    └────┬────┘    └────────┘  │  │
│  │         │               │                │               │ (if fails)          │  │
│  │         │               │                │               └──────▶ loop back    │  │
│  │         ▼               ▼                ▼                                      │  │
│  │    ┌────────────────────────────────────────────────────────────────────────┐  │  │
│  │    │                         AGENT TOOLS                                     │  │  │
│  │    │  • extract_keywords    • search_all_sources    • score_relevance       │  │  │
│  │    │  • identify_subtopics  • ingest_to_rag         • extract_findings      │  │  │
│  │    │  • generate_queries    • deduplicate_papers    • generate_synthesis    │  │  │
│  │    └────────────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
┌──────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│  SMART MODEL ROUTER  │  │     RAG PIPELINE        │  │   EXTERNAL APIS         │
│  ┌────────────────┐  │  │  ┌─────────────────┐    │  │  ┌─────────────────┐    │
│  │ Task Complexity│  │  │  │ Qdrant Vector   │    │  │  │ arXiv API       │    │
│  │   Analysis     │  │  │  │    Store        │    │  │  └─────────────────┘    │
│  └───────┬────────┘  │  │  └────────┬────────┘    │  │  ┌─────────────────┐    │
│          ▼           │  │           │             │  │  │ Semantic Scholar│    │
│  ┌────────────────┐  │  │  ┌────────▼────────┐    │  │  └─────────────────┘    │
│  │ Budget-Aware   │  │  │  │ Hybrid Search   │    │  └─────────────────────────┘
│  │   Routing      │  │  │  │ (Dense + BM25)  │    │
│  └───────┬────────┘  │  │  └────────┬────────┘    │
│          ▼           │  │           │             │
│  ┌────────────────┐  │  │  ┌────────▼────────┐    │
│  │ Gemini Models  │  │  │  │    Reranker     │    │
│  │ • flash-lite   │  │  │  │  (Cross-Encoder)│    │
│  │ • flash        │  │  │  └─────────────────┘    │
│  │ • pro          │  │  └─────────────────────────┘
│  └────────────────┘  │
└──────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA & CACHE LAYER                                   │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │     PostgreSQL       │  │       Redis          │  │      Qdrant                │  │
│  │  • Users & Auth      │  │  • LLM Response Cache│  │  • Paper Embeddings        │  │
│  │  • Projects          │  │  • Embedding Cache   │  │  • Semantic Search Index   │  │
│  │  • Paper References  │  │  • Session Store     │  │  • Collection Management   │  │
│  │  • Usage Tracking    │  │  • Pub/Sub Messages  │  │                            │  │
│  └──────────────────────┘  └──────────────────────┘  └────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer              | Technology                                       |
| ------------------ | ------------------------------------------------ |
| **Frontend**       | React 19, TypeScript, Tailwind CSS, MUI, Zustand |
| **Backend**        | FastAPI, Celery, SQLAlchemy                      |
| **AI/ML**          | Google Gemini, LangGraph, Qdrant                 |
| **Database**       | PostgreSQL, Redis                                |
| **Infrastructure** | Docker, Render, Vercel                           |

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Google Gemini API key

### 1. Clone & Configure

```bash
git clone https://github.com/sunilbishnoi1/scholar-agent.git
cd scholar-agent

# Create environment file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Start Services

```bash
docker-compose up -d
```

This starts:

- **Backend API** → http://localhost:8000
- **PostgreSQL** → localhost:5432
- **Redis** → localhost:6379
- **Qdrant** → http://localhost:6333

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at http://localhost:5173

## 📁 Project Structure

```
scholar-agent/
├── backend/
│   ├── agents/           # LangGraph agents & orchestrator
│   │   ├── orchestrator.py
│   │   ├── planner_agent.py
│   │   ├── analyzer_agent.py
│   │   └── synthesizer_agent.py
│   ├── rag/              # RAG pipeline
│   │   ├── vector_store.py
│   │   ├── hybrid_search.py
│   │   └── embeddings.py
│   ├── cache/            # Redis caching
│   ├── realtime/         # WebSocket manager
│   └── tests/            # Test suite
├── frontend/
│   └── src/
│       ├── components/   # React components
│       ├── pages/        # Route pages
│       ├── hooks/        # Custom hooks
│       └── store/        # Zustand stores
└── docs/                 # Documentation
```

## 🧪 Testing

```bash
# Backend tests
cd backend
source venv/Scripts/activate  # Windows
pytest tests/ -v --cov=. --cov-report=term-missing

# Frontend tests
cd frontend
npm test
```

## 📖 API Endpoints

| Method | Endpoint                   | Description             |
| ------ | -------------------------- | ----------------------- |
| `POST` | `/api/auth/register`       | Register new user       |
| `POST` | `/api/auth/token`          | Get JWT token           |
| `POST` | `/api/projects`            | Create research project |
| `GET`  | `/api/projects/{id}`       | Get project details     |
| `POST` | `/api/projects/{id}/start` | Start literature review |
| `WS`   | `/ws/projects/{id}/stream` | Real-time updates       |

Full API docs at http://localhost:8000/docs

## ⚙️ Configuration

| Variable         | Description                  | Default                  |
| ---------------- | ---------------------------- | ------------------------ |
| `GEMINI_API_KEY` | Google Gemini API key        | Required                 |
| `DATABASE_URL`   | PostgreSQL connection string | `postgresql://...`       |
| `REDIS_URL`      | Redis connection string      | `redis://localhost:6379` |
| `QDRANT_URL`     | Qdrant server URL            | `http://localhost:6333`  |
| `JWT_SECRET`     | Secret for JWT tokens        | Required                 |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<p align="center">
  Made with ❤️ by <a href="https://github.com/sunilbishnoi1">Sunil Bishnoi</a>
</p>
