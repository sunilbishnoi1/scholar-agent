# Scholar Agent: AI-Powered Research Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, multi-agent platform designed to automate the academic literature review process. Scholar Agent transforms a simple research question into a fully synthesized report, complete with identified research gaps, in a fraction of the time it takes to do manually.

**Live Demo:** [**https://scholar-agent.vercel.app/**](https://scholar-agent.vercel.app/)  **OR**  [**https://scholaragent.dpdns.org/**](https://scholaragent.dpdns.org/)


## 🧑‍💻 Author Information

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


## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
