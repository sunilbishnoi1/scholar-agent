# Scholar Agent: AI-Powered Research Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, multi-agent platform designed to automate the academic literature review process. Scholar Agent transforms a simple research question into a fully synthesized report, complete with identified research gaps, in a fraction of the time it takes to do manually.

**Live Demo:** [**https://scholaragent.dpdns.org/**](https://scholaragent.dpdns.org/)  **OR**  [**https://scholar-agent.vercel.app/**](https://scholar-agent.vercel.app/)


## 🧑‍💻 Author Information

This project was built as part of the AI Agent assignment.

*   **Name:** Sunil Bishnoi
*   **Roll Number:** B23ME1072
*   **Department:** Mechanical Engineering
*   **University:** Indian Institute of Technology, Jodhpur
*   **Email:** b23me1072@iitj.ac.in



## The Problem

For students, academics, and researchers, conducting a thorough literature review is a foundational yet incredibly time-consuming task, often taking weeks or even months. The process involves identifying relevant papers, meticulously analyzing each one, and synthesizing the findings to discover trends and research gaps. This manual effort is a significant bottleneck in the innovation pipeline.

## The Solution: Scholar Agent

Scholar Agent tackles this challenge head-on. By leveraging a multi-agent AI system, it can **reason, plan, and execute** the entire literature review workflow. Users simply provide a research topic and question, and the agent handles the rest, delivering a comprehensive report directly to the user's inbox. This frees up researchers to focus on what truly matters: pushing the boundaries of knowledge.

## ✨ Key Features

*   🤖 **Multi-Agent AI System:** A collaboration of three specialized agents (Planner, Analyzer, Synthesizer) that mimic the human research workflow.
*   🔎 **Research Gap Identification:** The Synthesizer agent is specifically designed to compare findings and highlight contradictions and limitations, making it easy to find novel research opportunities.
*   📊 **Real-Time Progress Tracking:** A dynamic user interface provides live updates on the agent's status, from searching for papers to analyzing and synthesizing the final report.
*   📚 **Integrated Academic Databases:** Retrieves the latest research from trusted sources like **arXiv** and **Semantic Scholar**.
*   📧 **Email & Document Delivery:** The final, synthesized literature review is automatically delivered to your **email** and can be exported as a **PDF or DOCX** file from the project page.
*   🔐 **Full User Authentication:** Secure user registration and login system to manage your personal research projects.
*   📱 **Progressive Web App (PWA):** Installable on both desktop and mobile devices for a native, app-like experience.
*   📄 **Access to Analyzed Papers:** View all the source papers used for your report, complete with their relevance scores and direct links to the original document.

## 🏗️ System Architecture

The platform is built on a modern, decoupled client-server architecture, ensuring scalability, maintainability, and a responsive user experience. The entire backend infrastructure, including the API, background worker, database, and message broker, is deployed on **Render**, while the frontend is served globally via **Vercel's** edge network.

```
+------------------+      +------------------+      +---------------------+
|   User Browser   |----->|  Vercel (React)  |<---->|   Render (FastAPI)  |
|      (PWA)       |      |    Frontend      |      |     Backend API     |
+------------------+      +------------------+      +----------+----------+
                                                               |
                                                               v
                                                      +-----------------+
                                                      | Render (Redis)  |
                                                      |   Task Broker   |
                                                      +--------+--------+
                                                               | (Queues Tasks)
                                                               v
+----------------------+      +------------------------+      +----------------------+
|  Google Gemini API   |<-----| Render (Celery Worker) |----->|  PostgreSQL DB       |
+----------------------+      |   Background Agent     |      | (Render)             |
| arXiv / Sem. Scholar |<-----|                        |<---->|                      |
+----------------------+      +------------------------+      +----------------------+
|    Brevo Email API   |<-----|                        |
+----------------------+      +------------------------+

```

## 🛠️ Tech Stack

| Category              | Technology                                                                                           |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| **Frontend**          | **React**, **TypeScript**, **Vite**, Zustand, React Query, Material-UI, Tailwind CSS                   |
| **Backend**           | **Python**, **FastAPI**, **Celery**, SQLAlchemy                                                          |
| **Database & Cache**  | **PostgreSQL** (Client-Server Database), **Redis** (Celery Broker)                                       |
| **AI & External APIs**| **Google Gemini API**, arXiv API, Semantic Scholar API, **Brevo** (Email)                               |
| **Deployment**        | **Vercel** (Frontend), **Render** (Backend API, Celery Worker, PostgreSQL, Redis), **Docker** (Dev) |

## 🚀 Getting Started: Local Development

To run this project locally, follow these steps.

### Prerequisites

*   Node.js (v18 or later)
*   Python (v3.9 or later)
*   Docker and Docker Compose
*   A code editor like VS Code

### 1. Clone the Repository

```bash
git clone https://github.com/sunilbishnoi1/scholar-agent.git
cd scholar-agent
```

### 2. Backend Setup (Docker)

The entire backend environment (FastAPI, PostgreSQL, Redis, Celery) is containerized for easy setup.

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Create an environment file:**
    Create a file named `.env` in the `backend` directory and add the following environment variables.

    ```env
    # Generate a secret key with: openssl rand -hex 32
    SECRET_KEY="your_super_secret_key"

    # API Keys
    GEMINI_API_KEY="your_google_gemini_api_key"
    BREVO_API_KEY="your_brevo_api_key"
    BREVO_SENDER_EMAIL="your_verified_sender_email@example.com"

    # These are the default values used in docker-compose.yml.
    # You don't need to change them for local development.
    DATABASE_URL="postgresql://user:password@db:5432/scholaragentdb"
    REDIS_URL="redis://redis:6379/0"
    ```

3.  **Build and run the containers:**
    ```bash
    docker-compose up --build
    ```
    The FastAPI server will be available at `http://localhost:8000`.

### 3. Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    # From the root directory
    cd frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Create an environment file:**
    Create a file named `.env.local` in the `frontend` directory and add the following variable:

    ```env
    VITE_API_BASE_URL=http://127.0.0.1:8000
    ```

4.  **Run the development server:**
    ```bash
    npm run dev
    ```
    The React application will be available at `http://localhost:5173`.


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.