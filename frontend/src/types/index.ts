// Type definitions for our data models

export interface User {
    id: string;
    email: string;
    name: string;
}

export interface ResearchProject {
    id: string;
    title: string;
    research_question: string;
    keywords: string[];
    subtopics: string[];
    status: 'planning' | 'searching' | 'analyzing' | 'synthesizing' | 'completed' | 'error' | 'error_no_papers_found' | 'created' | 'creating';
    created_at: string;
    paper_references: PaperReference[];
    agent_plans: AgentPlan[];
    total_papers_found: number;
}

export interface PaperReference {
    id: string;
    project_id: string;
    title: string;
    authors: string[];
    abstract: string;
    url: string;
    relevance_score: number;
}

export interface AgentPlan {
    id: string;
    project_id: string;
    agent_type: 'planner' | 'analyzer' | 'synthesizer';
    plan_steps: {
        step: string;
        status: string;
        output: {
            response: string | Record<string, unknown>;
        };
    }[];
    current_step: number;
    plan_metadata: Record<string, unknown>;
}

export interface ProjectCreate {
    title: string;
    research_question: string;
}