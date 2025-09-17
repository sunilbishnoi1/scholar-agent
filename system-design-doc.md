# System Design Document: Scholar Agent

## 1. Introduction

Scholar Agent is an AI-powered platform designed to automate and accelerate the academic literature review process. For students, academics, and researchers, conducting a thorough literature review is a foundational yet incredibly time-consuming task, often taking weeks or even months. The process involves identifying relevant papers, meticulously reading and analyzing each one, and synthesizing the findings to identify trends, contributions, and, most importantly, **research gaps**.

Scholar Agent addresses this challenge by employing a multi-agent AI system that can reason, plan, and execute the entire literature review workflow. Users simply provide a research topic and question, and the Scholar Agent handles the rest—from discovering relevant academic papers to delivering a fully synthesized report, complete with identified research gaps, directly to the **user's inbox**. This not only saves hundreds of hours of manual effort but also empowers researchers to focus on innovation and discovery.

## 2. System Architecture

The Scholar Agent platform is built on a modern, decoupled client-server architecture, supported by a robust background processing system to handle long-running, asynchronous AI tasks. This design ensures scalability, maintainability, and a responsive user experience.

The architecture consists of five primary layers:

1.  **Frontend (Client):** A Progressive Web App (PWA) built with React and Vite, responsible for all user interactions. It is deployed globally on **Vercel's** edge network.
2.  **Backend (Server):** A RESTful API server built with Python and FastAPI. This layer handles user authentication, project management, and orchestrates the AI agents. It is deployed as a Web Service on **Render**.
3.  **Background Processing:** A task queue system using Celery and Redis. This is the core of the AI workflow, executing the multi-agent processes asynchronously. It runs as a Background Worker on **Render**.
4.  **Data Persistence:** A PostgreSQL database for storing all application data, including user profiles, research projects, and analyzed papers. It is managed as a private service on **Render**.
5.  **External Services:** A suite of third-party APIs that provide the intelligence and data for the platform, including Google Gemini, arXiv, Semantic Scholar, and Brevo for email delivery.

### Architectural Diagram

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

## 3. Data Design

The data model is designed relationally to ensure data integrity, consistency, and efficient querying. We chose **PostgreSQL** as our database, deployed on Render.

### Why PostgreSQL?

*   **Reliability & ACID Compliance:** Guarantees that transactions are processed reliably, which is critical for managing user data and project states.
*   **Scalability:** It can handle a high volume of simultaneous connections and write operations, which is essential as the user base grows and more research projects are processed concurrently.
*   **Rich Feature Set:** Excellent support for JSON data types, allowing us to store semi-structured data like agent plan steps and keywords flexibly within a relational structure.
*   **Managed Service:** Using Render's managed PostgreSQL simplifies database administration, backups, and scaling.

### Database Schema

The schema is centered around four main models: `User`, `ResearchProject`, `PaperReference`, and `AgentPlan`.

*   **`User`**: Stores user authentication and profile information.
    *   `id` (PK, String/UUID)
    *   `email` (String, Unique)
    *   `name` (String)
    *   `hashed_password` (String)

*   **`ResearchProject`**: Represents a single literature review task initiated by a user.
    *   `id` (PK, String/UUID)
    *   `user_id` (FK to `User`)
    *   `title` (String)
    *   `research_question` (Text)
    *   `keywords` (JSON): A list of search terms generated by the Planner Agent.
    *   `subtopics` (JSON): A list of subtopics for structuring the review.
    *   `status` (String): Tracks the project's current state (e.g., `created`, `searching`, `analyzing`, `completed`, `error`).
    *   `total_papers_found` (Integer): The number of papers discovered, used for progress tracking on the frontend.

*   **`PaperReference`**: Stores metadata for each academic paper discovered and analyzed for a project.
    *   `id` (PK, String/UUID)
    *   `project_id` (FK to `ResearchProject`)
    *   `title` (String)
    *   `authors` (JSON)
    *   `abstract` (Text)
    *   `url` (String): Direct link to the paper source.
    *   `relevance_score` (Float): A score from 0-100 generated by the Analyzer Agent.

*   **`AgentPlan`**: A log of the steps and outputs generated by each agent for a given project.
    *   `id` (PK, String/UUID)
    *   `project_id` (FK to `ResearchProject`)
    *   `agent_type` (String): The agent responsible (e.g., `analyzer`, `synthesizer`).
    *   `plan_steps` (JSON): Detailed log of agent actions and their results.

## 4. Component Breakdown

### Frontend (Client-Side)

The frontend is a single-page application (SPA) built with **React** and **TypeScript**, bootstrapped with **Vite**.

*   **UI Framework:** A hybrid approach using **Material-UI (MUI)** for its robust component library and **Tailwind CSS** for rapid, utility-first styling and customization. This combination delivers a clean, modern, and responsive UI/UX.
*   **State Management:** **Zustand** is used for simple, boilerplate-free global state management, primarily for handling user authentication (`authStore`) and the list of research projects (`projectStore`).
*   **Data Fetching & Caching:** **TanStack React Query** manages all server-state. It simplifies data fetching, caching, and synchronization, providing a better user experience by reducing loading times and handling background updates.
*   **API Communication:** An **Axios** client is configured to communicate with the backend REST API, with interceptors to automatically attach the JWT authentication token to requests.
*   **Progressive Web App (PWA):** The `vite-plugin-pwa` is used to make the application installable on desktop and mobile devices, providing an app-like experience and enabling future offline capabilities.

### Backend (Server-Side)

The backend is a high-performance REST API built with **Python** and the **FastAPI** framework.

*   **API Style (REST):** REST was chosen for its simplicity, statelessness, and broad compatibility. It provides a clear and conventional way for the frontend to interact with the server.
*   **Authentication:** User authentication is implemented using **JWT (JSON Web Tokens)**. The backend provides `/register` and `/token` endpoints. Protected routes require a valid `Bearer` token, which is decoded and verified on each request.
*   **ORM:** **SQLAlchemy** is used as the Object-Relational Mapper, providing a robust and pythonic way to interact with the PostgreSQL database.
*   **Asynchronous Task Processing:** This is the most critical component of the backend architecture.
    *   **Celery:** A powerful distributed task queue used to run the long-running literature review process in the background. When a user starts a review, the API creates a task and places it on the queue, immediately returning a response to the user.
    *   **Redis:** Acts as the message broker for Celery, holding the queue of tasks to be executed by the Celery workers. It also serves as the result backend.
*   **Multi-Agent Collaboration:** The core logic is structured as a pipeline of collaborating agents:
    1.  **Planner Agent:** Triggered first. It uses the **Google Gemini API** to analyze the user's research question and generate a strategic plan, consisting of precise keywords and subtopics.
    2.  **Paper Retriever:** A utility component that takes the keywords and queries external academic APIs (**arXiv** and **Semantic Scholar**) to find relevant papers.
    3.  **Analyzer Agent:** This agent runs in a loop for each paper found. It uses the Gemini API to "read" the paper's abstract, extract key findings, methodologies, and limitations, and assign a relevance score.
    4.  **Synthesizer Agent:** The final agent in the pipeline. It takes the structured analyses from all papers and uses the Gemini API to weave them into a coherent literature review, highlighting themes and identifying research gaps.
*   **Email Integration:** Upon successful completion of the Synthesizer agent's task, the backend uses the **Brevo API** to send the final report directly to the user's registered email address.
*   **Containerization:** **Docker** was used for local development to create a consistent and reproducible environment, ensuring that all services (API, database, Redis) work together seamlessly. This container-first approach aligns perfectly with deployment on modern cloud platforms like Render.

## 5. Chosen Technologies & Reasoning

| Category          | Technology                                   | Reason for Choice                                                                                                                                                             |
| ----------------- | -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend**      | React, TypeScript, Vite                      | Industry-standard for building modern, fast, and maintainable user interfaces. Vite provides a superior development experience.                                                 |
|                   | Zustand, React Query                         | Lightweight yet powerful state management and server-state synchronization, reducing boilerplate and improving performance.                                                       |
|                   | **Deployment: Vercel**                       | Best-in-class platform for hosting frontend applications, offering a global CDN, seamless CI/CD from GitHub, and zero-configuration deployments.                                  |
| **Backend**       | Python, FastAPI                              | Python's rich ecosystem for AI/ML is essential. FastAPI provides extremely high performance (on par with NodeJS/Go), automatic API docs, and modern asynchronous capabilities. |
|                   | Celery, Redis                                | The go-to solution in the Python world for reliable, scalable background task processing. Crucial for handling long-running AI jobs without blocking the API or the user.     |
|                   | PostgreSQL, SQLAlchemy                       | A robust, production-grade relational database paired with a powerful ORM for data integrity and maintainability.                                                               |
|                   | **Deployment: Render**                       | A unified cloud platform that simplifies deploying all backend components (web service, worker, database, Redis) with private networking, auto-scaling, and Git-based deploys. |
| **AI & Data**     | Google Gemini API                            | A powerful, state-of-the-art Large Language Model that provides the core reasoning capabilities for all the AI agents at a competitive cost.                                  |
|                   | arXiv & Semantic Scholar APIs                | Provide free, programmatic access to millions of academic papers, forming the essential knowledge base for the literature search.                                               |
| **Communication** | Brevo                   | A reliable and developer-friendly transactional email API for delivering the final synthesized reports to users, a key feature of the platform.                                |

## 6. Originality & Social Impact

**Originality:** Scholar Agent's novelty lies in its practical application of a multi-agent AI architecture to a specific, high-value academic workflow. While general-purpose AI assistants exist, this platform is purpose-built for the literature review process. It doesn't just find papers; it reads, analyzes, and synthesizes them to generate new insights, specifically focusing on the critical task of **identifying research gaps**. The seamless integration of planning, analysis, and synthesis into a single, automated pipeline is a unique and powerful offering. The user interface, which provides real-time status updates on the agent's progress, further enhances the user experience and transparency of the AI's work.

**Social Impact:** The platform has the potential to significantly impact the academic and research communities:

*   **Accelerating Scientific Discovery:** By reducing the time for a literature review from months to minutes, researchers can move faster from ideation to experimentation, accelerating the overall pace of innovation.
*   **Democratizing Research:** It provides a powerful tool that was previously unavailable to under-resourced institutions or independent researchers, leveling the playing field.
*   **Educational Tool:** It can serve as an educational aid for graduate students, helping them understand the structure of a good literature review and quickly get up to speed on a new field of study.
*   **Combating Information Overload:** In an era where thousands of papers are published daily, Scholar Agent provides a crucial tool to navigate this flood of information and pinpoint the most relevant and impactful studies.