# 📚 Scholar Agent — Next-Gen Autonomous Multi-Agent Research Assistant

> Autonomous multi-agent scientific reasoning system for deep, full-text comparative literature reviews, cross-paper evidence extraction, citation auditing, and actionable research gap discovery.

[![CI/CD](https://github.com/sunilbishnoi1/scholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sunilbishnoi1/scholar-agent/actions)
[![codecov](https://codecov.io/gh/sunilbishnoi1/scholar-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/sunilbishnoi1/scholar-agent)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![React 19](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/Orchestrator-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Executive Overview

**Scholar Agent** is a production-grade autonomous multi-agent research platform designed for PhD candidates, researchers, and R&D engineers. Unlike traditional search engines or single-prompt LLM wrappers that only summarize short paper abstracts, Scholar Agent runs a **bounded hierarchical supervisor DAG (StateGraph)** across 7 specialized agents to acquire open-access full-text PDFs, extract LaTeX formulas and benchmark tables, pack section-aware contexts, synthesize thematic literature reviews, map methodological debates, identify unaddressed research gaps, and mathematically verify every citation claim via **Natural Language Inference (NLI)**.

---

## ✨ Core Features & Capabilities

- 🤖 **Bounded Hierarchical Multi-Agent DAG** — Orchestrated by LangGraph with a Supervisor/Principal Investigator that coordinates 7 specialized agents with bounded adversarial refinement loops (max 1–2 cycles) to prevent token explosion and guarantee deterministic completion.
- 📄 **Full-Text Ingestion & Open-Access Cascade** — Multi-tier never-fail resolution cascade (**Unpaywall → arXiv → Semantic Scholar → OpenAlex / PubMed → Structured Abstract Fallback**) with **Docling / Marker** extracting structured Markdown, LaTeX math, and multi-column tables.
- 🧩 **Section-Aware Context Packing** — Hierarchical sectioning (`Methodology`, `Results`, `Limitations`, `Tables`) eliminating the "Lost-in-the-Middle" attention degradation and optimizing LLM token budgets.
- 📊 **Structured Cross-Paper Evidence Matrix** — Extracts uniform comparative schemas across all acquired papers (*Problem Formulation*, *Proposed Architecture*, *Benchmark Datasets*, *Primary Quantitative Metrics*, and *Author-Stated Limitations*).
- ⚔️ **Empirical & Methodological Debate Mapping** — Uncovers conflicting findings and opposing paradigms across papers (e.g., *Perspective A vs Perspective B* on benchmark performance) with critical evaluations.
- 🎯 **Actionable Research Gap Discovery** — Synthesizes high-impact unaddressed research gaps directly grounded in bibliography with recommended methodology directions.
- 🛡️ **Zero-Hallucination NLI Citation Grounding** — Deconstructs synthesis text into atomic propositions and verifies them against source section chunks using structured LLM Natural Language Inference (`ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`) with deep citation anchors (e.g., `[ref_1#sec3]`).
- ⚡ **Real-Time WebSocket Streaming** — Live pipeline visualization powered by Celery async background workers and Redis Pub/Sub, broadcasting stage progress, active agents, and terminal logs.
- 🧠 **Lean 2-Tier Memory & Relational Caching** — Scoped LangGraph working memory blackboard + PostgreSQL `paper_cache` for cross-project DOI deduplication + project-scoped Qdrant hybrid vector store (Dense embeddings + BM25 keyword search + Cross-Encoder reranker).
- 📥 **Multi-Format Academic Export Engine** — Export complete research intelligence reports to **Interactive Web**, **Markdown**, **PDF**, **DOCX**, and **BibTeX (.bib)**.


## 🏗️ System Architecture

### End-to-End System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FRONTEND LAYER                                            │
│                       React 19 + TypeScript + Vite + Tailwind CSS + MUI + Zustand               │
│                                                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │   Synthesis & Prose     │  │   Cross-Paper Evidence   │  │   Empirical & Methodological   │  │
│  │   Thematic Anchors      │  │      Matrix Table        │  │          Debate Cards          │  │
│  └─────────────────────────┘  └──────────────────────────┘  └────────────────────────────────┘  │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │ Actionable Research Gaps│  │  BibTeX / PDF Exporter   │  │ Real-Time Agent Journey & Logs │  │
│  │  & Grounded Methodology │  │  Multi-Format Publishing │  │    WebSocket Live Visualizer   │  │
│  └─────────────────────────┘  └──────────────────────────┘  └────────────────────────────────┘  │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │ REST API / WebSocket Stream
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    BACKEND API & WORKER LAYER                                   │
│                                   FastAPI + Celery + SQLAlchemy                                 │
│                                                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │  JWT Auth & User Quotas │  │ Project & Report Manager │  │ Redis Pub/Sub Event Broadcast  │  │
│  │   /api/auth, /api/user  │  │       /api/projects      │  │        /ws/projects/stream     │  │
│  └─────────────────────────┘  └──────────────────────────┘  └────────────────────────────────┘  │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │ Dispatches Task
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 LANGGRAPH SUPERVISOR DAG ORCHESTRATION                          │
│                                                                                                 │
│                ┌─────────────────────────────────────────────────────────────┐                  │
│                │ 1. Supervisor / Principal Investigator (Goal Stack & DAG)   │                  │
│                └──────────────────────────────┬──────────────────────────────┘                  │
│                                               ▼                                                 │
│                ┌─────────────────────────────────────────────────────────────┐                  │
│                │ 2. Literature Explorer (OpenAlex, arXiv, S2, PubMed Search) │                  │
│                └──────────────────────────────┬──────────────────────────────┘                  │
│                                               ▼                                                 │
│                ┌─────────────────────────────────────────────────────────────┐                  │
│                │ 3. Full-Text Ingestion (OA Cascade, Docling/Marker, Tables) │                  │
│                └──────────────────────────────┬──────────────────────────────┘                  │
│                                               ▼                                                 │
│                ┌─────────────────────────────────────────────────────────────┐                  │
│                │ 4. Evidence Matrix Builder (Problem, Method, Benchmarks)    │                  │
│                └──────────────────────────────┬──────────────────────────────┘                  │
│                                               ▼                                                 │
│                ┌─────────────────────────────────────────────────────────────┐                  │
│                │ 5. Thematic Synthesizer (Thematic Prose, Debates, Gaps)     │◀───────────┐     │
│                └──────────────────────────────┬──────────────────────────────┘            │     │
│                                               ▼                                           │     │
│                ┌─────────────────────────────────────────────────────────────┐            │     │
│                │ 6. Adversarial Critic & Methodologist (Rigor Scoring)       │──[score<75]┘     │
│                └──────────────────────────────┬──────────────────────────────┘ (Max 2 Cycles)   │
│                                               ▼ [score >= 75 or max loops]                      │
│                ┌─────────────────────────────────────────────────────────────┐                  │
│                │ 7. Citation Auditor & Fact-Checker (NLI Claim Verification) │                  │
│                └──────────────────────────────┬──────────────────────────────┘                  │
│                                               ▼                                                 │
│                ┌─────────────────────────────────────────────────────────────┐                  │
│                │ Finalizer ──▶ Structured Deliverable Report & Pydantic Sync │                  │
│                └─────────────────────────────────────────────────────────────┘                  │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
┌────────────────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────────────┐
│      HIGH-CAPACITY LLMs        │ │    FULL-TEXT RAG ENGINE   │ │    EXTERNAL ACADEMIC APIs      │
│  ┌──────────────────────────┐  │ │  ┌─────────────────────┐  │ │  ┌──────────────────────────┐  │
│  │ Google Gemini 3.5        │  │ │  │ Project Vector DB   │  │ │  │ OpenAlex API (Citations) │  │
│  │ (Flash & Lite)           │  │ │  │ (Qdrant / pgvector) │  │ │  └──────────────────────────┘  │
│  └──────────────────────────┘  │ │  └──────────┬──────────┘  │ │  ┌──────────────────────────┐  │
│  ┌──────────────────────────┐  │ │             ▼             │ │  │ Semantic Scholar API     │  │
│  │ DeepSeek-V4-flash        │  │ │  ┌─────────────────────┐  │ │  └──────────────────────────┘  │
│  │ (Reasoning & Critique)   │  │ │  │ Hybrid Search       │  │ │  ┌──────────────────────────┐  │
│  └──────────────────────────┘  │ │  │ (Dense + BM25)      │  │ │  │ arXiv API / Unpaywall    │  │
│  ┌──────────────────────────┐  │ │  └──────────┬──────────┘  │ │  └──────────────────────────┘  │
│  │ Pydantic Structured      │  │ │             ▼             │ │  ┌──────────────────────────┐  │
│  │ Outputs (Schema-Enforced)│  │ │  ┌─────────────────────┐  │ │  │ PubMed / Europe PMC      │  │
│  └──────────────────────────┘  │ │  │ Cross-Encoder       │  │ │  └──────────────────────────┘  │
│                                │ │  │ Reranker (Precision)│  │ └────────────────────────────────┘
│                                │ │  └─────────────────────┘  │
└────────────────────────────────┘ └───────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    LEAN 2-TIER DATA & CACHE LAYER                               │
│                                                                                                 │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────────┐  ┌───────────────┐  │
│  │         PostgreSQL Database      │  │           Redis Cache            │  │ Qdrant Vector │  │
│  │  • Users, Projects, Quotas       │  │  • Real-Time Pub/Sub Streaming   │  │ • Section     │  │
│  │  • Research Reports & Gaps       │  │  • Fast LLM Response Cache       │  │   Embeddings  │  │
│  │  • Evidence Matrix Entries       │  │  • Session State & Task Queue    │  │ • Hierarchical│  │
│  │  • Global `paper_cache` (DOI dedup) │                              │  │   Search Index│  │
│  └──────────────────────────────────┘  └──────────────────────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### LangGraph Multi-Agent Workflow DAG

```mermaid
flowchart TD
    Start([Start Literature Review]) --> Supervisor[1. Supervisor / Principal Investigator]
    Supervisor --> Discovery[2. Literature Discovery Agent]
    Discovery -->|Candidate Paper DOIs & URLs| Ingestion[3. Full-Text Ingestion Specialist]
    Ingestion -->|Structured Markdown, Sections & Tables| Blackboard[(In-Flight Working Memory Blackboard)]
    Blackboard --> MatrixBuilder[4. Evidence Matrix Builder]
    MatrixBuilder --> Synthesizer[5. Thematic Synthesizer]
    Synthesizer --> Critic[6. Adversarial Critic & Methodologist]
    
    Critic -->|Quality Score < 75 & Iteration < 2| Synthesizer
    Critic -->|Quality Score >= 75 or Max Iterations Reached| Auditor[7. Citation Auditor & Fact-Checker]
    
    Auditor --> Finalizer[Final Deliverable Report Assembly]
    Finalizer --> End([Completed Research Intelligence Deliverable])

    classDef agent fill:#18181B,stroke:#FFB900,stroke-width:2px,color:#F4F4F5;
    classDef memory fill:#1E293B,stroke:#00F5C8,stroke-width:2px,color:#F4F4F5;
    classDef terminal fill:#09090B,stroke:#818CF8,stroke-width:2px,color:#F4F4F5;

    class Supervisor,Discovery,Ingestion,MatrixBuilder,Synthesizer,Critic,Auditor,Finalizer agent;
    class Blackboard memory;
    class Start,End terminal;
```

---

## 🤖 The 7 Autonomous Agents

| # | Agent Name | Role | Behavioral Contract & Autonomous Scope |
|---|------------|------|---------------------------------------|
| **1** | **Supervisor / Principal Investigator** | DAG Orchestrator & Goal Stack Manager | Formulates research inquiry DAG, allocates token budgets, tracks agent goals, and terminates refinement loops. |
| **2** | **Literature Explorer** | Multi-Source Discovery Agent | Queries OpenAlex, Semantic Scholar, arXiv, and PubMed with dynamic Boolean expansion and citation graph snowballing. |
| **3** | **Full-Text Ingestion Specialist** | PDF Resolution & Section Parser | Executes the multi-tier open-access cascade, extracts structured Markdown, formulas (LaTeX), and benchmark tables via Docling/Marker. |
| **4** | **Evidence Matrix Builder** | Tabular Comparative Analyst | Extracts uniform schemas: Problem, Architecture, Benchmark Datasets, Metrics, and Limitations across all papers. |
| **5** | **Thematic Synthesizer** | Scientific Prose Architect | Generates dense thematic reviews, comparative syntheses, opposing scientific schools of thought, and unaddressed research gaps. |
| **6** | **Adversarial Critic** | Peer Reviewer & Quality Evaluator | Evaluates empirical rigor, statistical validity, and baseline coverage. Triggers bounded iterative refinement if score < 75. |
| **7** | **Citation Auditor** | NLI Fact-Checker & Grounding Engine | Deconstructs text into atomic claims and verifies grounding against source chunks using structured NLI (`ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`). |

---

## 🛠️ Modern Tech Stack

| Layer | Technology | Description |
|-------|------------|-------------|
| **Frontend UI** | **React 19**, **TypeScript**, **Tailwind CSS**, **MUI 6**, **Zustand**, **TanStack Query** | High-performance reactive dashboard with dark mode, live WebSocket stream, and interactive tables |
| **Backend API** | **FastAPI**, **Celery**, **SQLAlchemy**, **Pydantic v2** | High-throughput asynchronous REST API & background task execution |
| **Agent Framework** | **LangGraph (StateGraph)** | Directed Acyclic Graph orchestration with state machines and conditional refinement edges |
| **AI / LLMs** | **Google Gemini 3.5 (Flash & Lite)**, **DeepSeek-V4-flash**, **Groq** | Native Pydantic structured outputs, large context reasoning (128K–1M+ tokens) |
| **Full-Text Ingestion** | **Docling**, **Marker**, **PyMuPDF**, **pdfminer.six** | Structure-preserving PDF parsing, table extraction, and LaTeX math OCR |
| **RAG & Vector Search** | **Qdrant Vector Database**, **BM25**, **FastEmbed**, **Cross-Encoder** | Hybrid semantic + lexical retrieval with reciprocal rank fusion and reranking |
| **Relational Storage** | **PostgreSQL (Neon serverless / Local)** | Relational persistence, JSONB document trees, and global `paper_cache` DOI store |
| **Cache & Realtime** | **Redis (Native / Memurai / Upstash)** | Pub/Sub event broadcaster, LLM response cache, and Celery task broker |
| **Deployment** | **Docker**, **Docker Compose**, **Render**, **Vercel** | Containerized microservice architecture ready for production cloud hosting |

## 🚀 Quick Start & Local Setup

You can run Scholar Agent either with **Docker Compose** or directly **Without Docker** using native processes on your machine.

---

### Option A: Local Setup Without Docker (Native Development & Manual Testing)

This approach runs the services natively on Windows, macOS, or Linux using local / cloud databases without requiring Docker.

#### 1. Prerequisites
- **Python**: 3.11+ (recommended with virtual environment)
- **Node.js**: 18+ and `npm`
- **Redis**: Native Windows Redis (in `./redis/redis-server.exe`), Memurai (`winget install Memurai.MemuraiDeveloper`), WSL Redis, or Cloud Upstash
- **Database**: Cloud PostgreSQL (e.g. Neon serverless) or local PostgreSQL
- **Vector Store**: Qdrant Cloud cluster or local binary

#### 2. Environment Configuration
Clone the repository and copy the environment template:
```bash
git clone https://github.com/sunilbishnoi1/scholar-agent.git
cd scholar-agent

cp .env.example .env
```
Edit `.env` with your API keys and connection URLs:
```env
# AI Providers
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENALEX_API_KEY=your_openalex_key_here

# Databases & Vector Store
DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# Feature Flags
ENABLE_REDIS=true
ENABLE_CELERY=true
```

#### 3. Start Redis Server
- **Windows (Standalone)**:
  ```powershell
  # Using the included standalone redis binary:
  & ".\redis\redis-server.exe" ".\redis\redis.windows.conf"
  ```
- **Windows (Memurai / Service)**:
  ```powershell
  winget install Memurai.MemuraiDeveloper
  net start Memurai
  ```
- **Linux / macOS / WSL**:
  ```bash
  redis-server
  ```

#### 4. Start the Backend API & Celery Worker

- **Terminal 1: FastAPI Backend Server**
  ```bash
  # Windows
  .venv\Scripts\activate
  cd backend
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

  # Linux / macOS
  source .venv/bin/activate
  cd backend
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
  ```
  API Docs will be live at: `http://localhost:8000/docs`

- **Terminal 2: Celery Background Worker**
  ```bash
  # Windows (Note: -P solo is required on Windows)
  .venv\Scripts\activate
  cd backend
  celery -A main.celery_app worker --loglevel=info -P solo

  # Linux / macOS
  source .venv/bin/activate
  cd backend
  celery -A main.celery_app worker --loglevel=info
  ```

#### 5. Start the Frontend Application

- **Terminal 3: React + Vite Frontend**
  ```bash
  cd frontend
  npm install
  npm run dev
  ```
  Frontend will be live at: `http://localhost:5173` (or `5174`)

#### 6. Run Live End-to-End Real API Verification (Terminal / CLI Test)
To verify the entire autonomous multi-agent pipeline with real API calls without opening the browser:
```bash
# Windows
.venv\Scripts\activate
cd backend
python test_real_literature_review.py
```
This tests real academic queries against arXiv/OpenAlex, PDF ingestion, Qdrant embeddings, Gemini analysis, Evidence Matrix extraction, Supervisor quality loops, and Citation Auditor fact-checking.

---

### Option B: Containerized with Docker Compose

If you have Docker Desktop installed, you can start all containerized services with a single command:

```bash
# 1. Start all containerized infrastructure & services
docker-compose up -d

# 2. Start Frontend
cd frontend
npm install
npm run dev
```

This boots:
- **Backend API** → `http://localhost:8000`
- **PostgreSQL** → `localhost:5432`
- **Redis** → `localhost:6379`
- **Qdrant** → `http://localhost:6333`
- **Celery Worker** & **Celery Beat**

---

## 📁 Project Structure

```
scholar-agent/
├── backend/
│   ├── agents/
│   │   ├── core/
│   │   │   ├── supervisor.py         # Supervisor Agent (LangGraph DAG Coordinator)
│   │   │   ├── discovery.py          # Literature Explorer (OpenAlex, S2, arXiv, PubMed)
│   │   │   ├── ingestion.py          # Full-Text Acquisition & Section/Table Parser
│   │   │   ├── matrix_builder.py     # Cross-Paper Evidence Matrix Constructor
│   │   │   ├── synthesizer.py        # Thematic Synthesis & Gap Discovery Specialist
│   │   │   ├── critic.py             # Adversarial Peer Reviewer & Methodologist
│   │   │   └── auditor.py            # Structured LLM NLI Citation & Fact-Checker
│   │   ├── blackboard.py             # Scoped In-Flight Agent State & Working Memory
│   │   ├── tools/
│   │   │   ├── academic_search.py    # Academic Database Search Engine
│   │   │   ├── oa_resolver.py        # Multi-Tier Open-Access Resolution Cascade
│   │   │   ├── pdf_parser.py         # Full-Text PDF, LaTeX Math & Table Parser
│   │   │   ├── citation_graph.py     # Citation Network Snowballing
│   │   │   └── fact_checker.py       # Atomic Proposition & NLI Entailment Verifier
│   │   ├── llm/
│   │   │   ├── base.py               # Unified BaseLLMClient interface
│   │   │   ├── gemini_provider.py    # Google Gemini 3.5 Flash / Lite Provider
│   │   │   ├── deepseek_provider.py  # DeepSeek-V4-flash Provider
│   │   │   └── structured_output.py  # Pydantic schema validation engine
│   │   └── schemas.py                # Pydantic ResearchReport, EvidenceMatrix, Paper schemas
│   ├── rag/
│   │   ├── vector_store.py           # Project-scoped Qdrant / pgvector store
│   │   ├── chunker.py                # Section-aware hierarchical chunker
│   │   ├── embeddings.py             # Vector embedding generation
│   │   └── hybrid_search.py          # BM25 + Dense vector hybrid search & reranking
│   ├── realtime/
│   │   ├── events.py                 # WebSocket event definitions & progress trackers
│   │   └── manager.py                # Redis Pub/Sub connection manager
│   ├── models/
│   │   └── database.py               # SQLAlchemy models (PaperCache, ResearchReport, etc.)
│   ├── auth.py                       # JWT authentication and user management
│   ├── db.py                         # Database engine and session factory
│   ├── main.py                       # FastAPI application & Celery background tasks
│   └── tests/                        # Comprehensive test suite
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── dashboard/
│       │   │   ├── AgentPipeline.tsx             # Live multi-agent journey & terminal log
│       │   │   ├── EvidenceMatrixTable.tsx       # Interactive cross-paper comparison matrix
│       │   │   ├── ThematicSections.tsx          # Synthesis prose with citation anchors
│       │   │   ├── ConflictingDebates.tsx        # Adversarial debate & divergence cards
│       │   │   ├── ResearchGapViewer.tsx         # Actionable research gaps & methods
│       │   │   └── MethodologyDistributionCard.tsx # Visual breakdown of research paradigms
│       │   └── common/                           # Reusable UI components
│       ├── pages/
│       │   ├── DashboardPage.tsx                 # User projects overview
│       │   ├── ProjectDetailsPage.tsx            # Multi-tab research intelligence report
│       │   └── LandingPage.tsx                   # Product landing & workflow introduction
│       ├── hooks/
│       │   └── useProjectStream.ts               # WebSocket real-time event listener
│       └── types/
│           └── index.ts                          # TypeScript interfaces matching backend models
└── docs/                                         # Architectural documentation & specifications
```

## 🧪 Testing

```bash
# Run Backend Test Suite
cd backend
pytest tests/ -v --cov=. --cov-report=term-missing

# Run Frontend Test Suite
cd frontend
npm test
```

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/register` | Register new user account |
| `POST` | `/api/auth/token` | Obtain JWT access token |
| `GET` | `/api/auth/me` | Fetch authenticated user profile & quota status |
| `POST` | `/api/projects` | Create a new research project with target query |
| `GET` | `/api/projects` | List all projects belonging to user |
| `GET` | `/api/projects/{id}` | Get project details, paper count, and status |
| `POST` | `/api/projects/{id}/start` | Launch autonomous multi-agent literature review |
| `POST` | `/api/projects/{id}/stop` | Cancel in-flight literature review task |
| `GET` | `/api/projects/{id}/report` | Retrieve final structured `ResearchReport` deliverable |
| `GET` | `/api/projects/{id}/matrix` | Retrieve structured `EvidenceMatrix` entries |
| `GET` | `/api/projects/{id}/gaps` | Retrieve identified `ResearchGaps` |
| `GET` | `/api/projects/{id}/export?format={fmt}` | Export report (`markdown`, `pdf`, `docx`, `bibtex`) |
| `WS` | `/ws/projects/{id}/stream` | Real-time WebSocket event stream for live agent progress |

Full OpenAPI specification is available at `http://localhost:8000/docs`.

## ⚙️ Configuration

| Variable | Description | Required / Default |
|----------|-------------|--------------------|
| `GEMINI_API_KEY` | Google Gemini API key (Primary engine for Gemini 3.5 Flash / Lite) | **Required** |
| `GROQ_API_KEY` | Groq API key (Optional secondary high-speed LLM provider) | Optional |
| `OPENALEX_API_KEY` | OpenAlex API key for enhanced query rate limits | Optional |
| `DATABASE_URL` | PostgreSQL connection URL (`postgresql://user:pass@host:port/db`) | **Required** |
| `REDIS_URL` | Redis connection URL (`redis://localhost:6379/0`) | `redis://localhost:6379/0` |
| `ENABLE_REDIS` | Toggle Redis caching and Pub/Sub streaming | `true` |
| `ENABLE_CELERY` | Toggle background Celery worker processing | `true` |
| `QDRANT_URL` | Qdrant vector database URL (Cluster endpoint or local) | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key (Required for Qdrant Cloud clusters) | Optional |
| `JWT_SECRET` | Secret key for JWT authentication token generation | **Required** |
| `BREVO_API_KEY` | Brevo API key for transactional review completion email alerts | Optional |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<p align="center">
  Crafted with precision for the global scientific research community by <a href="https://github.com/sunilbishnoi1">Sunil Bishnoi</a>
</p>

