export interface AgentData {
    name: string;
    role: string;
    description: string;
}

export type AgentId =
    | 'supervisor'
    | 'discovery'
    | 'ingestion'
    | 'matrix_builder'
    | 'synthesizer'
    | 'critic'
    | 'auditor'
    | 'planner'
    | 'retriever'
    | 'analyzer';

export interface DAGNode {
    id: AgentId;
    name: string;
    shortName: string;
    role: string;
    description: string;
    stageNumber: number;
    color: string;
}

export const DAG_NODES: DAGNode[] = [
    {
        id: 'supervisor',
        name: 'LangGraph Supervisor',
        shortName: 'Supervisor',
        role: 'Orchestrator DAG Coordinator',
        description: 'Dynamically routes execution state and manages refinement bounds',
        stageNumber: 1,
        color: '#818CF8', // Indigo
    },
    {
        id: 'discovery',
        name: 'Literature Discovery',
        shortName: 'Discovery',
        role: 'Multi-Source Academic Explorer',
        description: 'Executes parallel faceted search across OpenAlex, Semantic Scholar, and arXiv',
        stageNumber: 2,
        color: '#00F5C8', // Aurora Teal
    },
    {
        id: 'ingestion',
        name: 'Ingestion & PDF Parser',
        shortName: 'PDF Ingestion',
        role: 'Open-Access Resolver & Chunker',
        description: 'Resolves unpaywalled PDFs and extracts hierarchical section chunks',
        stageNumber: 3,
        color: '#38BDF8', // Sky Blue
    },
    {
        id: 'matrix_builder',
        name: 'Evidence Matrix Builder',
        shortName: 'Matrix Builder',
        role: 'Structured Extraction Engine',
        description: 'Extracts problem, method, dataset benchmarks, metrics, and limitations',
        stageNumber: 4,
        color: '#FBBF24', // Amber
    },
    {
        id: 'synthesizer',
        name: 'Thematic Synthesizer',
        shortName: 'Synthesizer',
        role: 'Scientific Prose Architect',
        description: 'Drafts dense thematic synthesis with citation anchors and debate maps',
        stageNumber: 5,
        color: '#34D399', // Emerald
    },
    {
        id: 'critic',
        name: 'Adversarial Critic',
        shortName: 'Critic',
        role: 'Quality Evaluator & Loop Controller',
        description: 'Scores draft against academic rigor rubric (loops if score < 75)',
        stageNumber: 6,
        color: '#F43F5E', // Rose
    },
    {
        id: 'auditor',
        name: 'Citation Auditor',
        shortName: 'Auditor',
        role: 'NLI Fact-Checking Verifier',
        description: 'Grounds propositions against source chunks via natural language inference',
        stageNumber: 7,
        color: '#A78BFA', // Purple
    },
];

export const agentColors: { [key: string]: string } = {
    'Supervisor': '#818CF8',
    'supervisor': '#818CF8',
    'Discovery': '#00F5C8',
    'discovery': '#00F5C8',
    'Ingestion': '#38BDF8',
    'ingestion': '#38BDF8',
    'Matrix Builder': '#FBBF24',
    'matrix_builder': '#FBBF24',
    'Synthesizer': '#34D399',
    'synthesizer': '#34D399',
    'Critic': '#F43F5E',
    'critic': '#F43F5E',
    'Auditor': '#A78BFA',
    'auditor': '#A78BFA',
    // Legacy colors
    'Planner Agent': '#00F5C8',
    'planner': '#00F5C8',
    'Retriever Agent': '#00B894',
    'retriever': '#00B894',
    'Analyzer Agent': '#FFB900',
    'analyzer': '#FFB900',
    'Quality Checker Agent': '#A1A1AA',
    'Synthesizer Agent': '#00B88D',
};
