// Type definitions for Scholar Agent data models (Scholar Agent v3.2)

export interface User {
    id: string;
    email: string;
    name: string;
}

export interface ResearchProject {
    id: string;
    user_id?: string;
    title: string;
    research_question: string;
    keywords: string[];
    subtopics: string[];
    status:
        | 'created'
        | 'creating'
        | 'planning'
        | 'searching'
        | 'analyzing'
        | 'synthesizing'
        | 'completed'
        | 'in_progress'
        | 'stopped'
        | 'error'
        | 'error_no_papers_found';
    created_at: string;
    paper_references: PaperReference[];
    agent_plans: AgentPlan[];
    total_papers_found: number;
    max_papers?: number;
    report?: ResearchReport | null;
    report_status?: string;
}

export interface PaperReference {
    id: string;
    project_id?: string;
    paper_id?: string;
    title: string;
    authors: string[];
    abstract?: string;
    url?: string;
    doi?: string;
    arxiv_id?: string;
    venue?: string;
    year?: number;
    citation_count?: number;
    relevance_score: number;
    is_full_text?: boolean;
}

export interface AgentPlan {
    id: string;
    project_id: string;
    agent_type: 'planner' | 'analyzer' | 'synthesizer' | 'supervisor' | 'discovery' | 'ingestion' | 'matrix_builder' | 'critic' | 'auditor';
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
    max_papers?: number;
}

// ============================================================================
// Pydantic v3.2 Research Report Component Schemas
// ============================================================================

export type ReportStatus = 'complete' | 'completed' | 'partial' | 'analysis_only' | 'error' | 'in_progress' | 'empty';

export interface EvidenceMatrixRow {
    id?: string;
    paper_id: string;
    title: string;
    authors: string[];
    year: number | null;
    methodology: string;
    methodology_type?: string;
    benchmark_dataset: string;
    dataset?: string;
    primary_metric: string;
    primary_limitation: string;
    limitations?: string[];
    is_full_text?: boolean;
    has_full_text?: boolean;
    confidence_score?: number;
    key_findings?: string[];
    performance_metrics?: Record<string, unknown>;
    created_at?: string | null;
    url?: string | null;
    doi?: string | null;
}

export interface ThematicSection {
    theme_id: string;
    title: string;
    synthesis_prose: string;
    key_takeaways: string[];
    cited_paper_ids: string[];
    content?: string;
    key_insight?: string;
}

export interface ConflictingDebate {
    topic: string;
    perspective_a: string;
    perspective_b: string;
    critical_evaluation: string;
}

export interface ResearchGapItem {
    id?: string;
    gap_id: string;
    title?: string;
    description: string;
    importance: 'high' | 'medium' | 'low';
    priority?: string;
    recommended_methodology: string;
    actionable_recommendations?: string[];
    grounding_paper_ids: string[];
    grounding_papers?: string[];
    created_at?: string | null;
}

export interface MethodologyDistribution {
    distribution: Record<string, number>;
    dominant_approach: string;
    trend_description: string;
}

export interface BibliographyItem {
    paper_id: string;
    title: string;
    authors: string[];
    year: number | null;
    venue?: string | null;
    doi?: string | null;
    pdf_url?: string | null;
    url?: string | null;
    citation_count?: number | null;
    is_full_text_analyzed?: boolean;
    bibtex?: string;
    source?: string;
}

export interface ReportMetadata {
    project_id?: string;
    user_id?: string;
    title: string;
    research_question: string;
    generated_at: string;
    pipeline_duration_seconds?: number;
    status?: ReportStatus | string;
    quality_score?: number;
    papers_analyzed_full_text?: number;
    total_citations?: number;
    total_papers_analyzed?: number;
    llm_calls_made?: number;
    tokens_consumed?: number;
    models_used?: string[];
    target_academic_level?: string;
    synthesis_version?: string;
}

export interface ResearchReport {
    metadata: ReportMetadata;
    title?: string;
    executive_summary: string;
    comparison_matrix?: EvidenceMatrixRow[];
    comparative_matrix?: EvidenceMatrixRow[];
    thematic_sections: ThematicSection[];
    sections?: ThematicSection[];
    conflicting_findings_and_debates?: ConflictingDebate[];
    conflicting_debates?: ConflictingDebate[];
    debates?: ConflictingDebate[];
    actionable_research_gaps?: ResearchGapItem[];
    actionable_gaps?: ResearchGapItem[];
    research_gaps?: ResearchGapItem[];
    methodology_overview: MethodologyDistribution;
    bibliography: BibliographyItem[];
    quality_score?: number;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ReportResponse {
    project_id: string;
    report: ResearchReport | null;
    report_status: string;
    message?: string;
}

export interface MatrixResponse {
    project_id: string;
    count: number;
    total: number;
    entries: EvidenceMatrixRow[];
    matrix?: EvidenceMatrixRow[];
}

export interface GapsResponse {
    project_id: string;
    count: number;
    total: number;
    gaps: ResearchGapItem[];
}

export interface SectionChunk {
    section_id?: string;
    title?: string;
    section_type?: string;
    text?: string;
    content?: string;
    page_start?: number;
    page_end?: number;
}

export interface PaperSectionsResponse {
    paper_id: string;
    doi?: string | null;
    arxiv_id?: string | null;
    s2_id?: string | null;
    title: string;
    authors: string[];
    year?: number | null;
    venue?: string | null;
    abstract?: string | null;
    is_full_text: boolean;
    sections: SectionChunk[];
    tables?: unknown[];
    source_url?: string | null;
}

// ============================================================================
// WebSocket Event Types (Scholar Agent v3.2 & Legacy)
// ============================================================================

export type EventType =
    | 'connected'
    | 'disconnected'
    // Standard v3.2 granular agent events
    | 'discovery_started'
    | 'paper_discovered'
    | 'pdf_parsed'
    | 'matrix_row_added'
    | 'thematic_draft_ready'
    | 'critic_verdict'
    | 'fact_checked'
    | 'pipeline_completed'
    | 'pipeline_error'
    | 'pipeline_stopped'
    // Legacy agent lifecycle events
    | 'agent_started'
    | 'agent_completed'
    | 'agent_error'
    | 'status'
    | 'progress'
    | 'log'
    | 'paper_found'
    | 'paper_analyzed'
    | 'complete'
    | 'error'
    | 'pong';

export interface AgentWebSocketEvent {
    type: EventType;
    agent?: string;
    project_id?: string;
    message?: string;
    progress?: number;
    data?: Record<string, unknown>;
    timestamp?: string;
}
