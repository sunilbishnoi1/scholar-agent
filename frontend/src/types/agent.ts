export interface AgentData {
    name: string;
    role: string;
    description: string;
}

export const agentColors: { [key: string]: string } = {
    'Planner Agent': '#00F5C8', // Aurora Teal 500
    'Retriever Agent': '#00B894', // Discovery Green 500
    'Analyzer Agent': '#FFB900', // Insight Amber 500
    'Quality Checker Agent': '#A1A1AA', // Obsidian 300
    'Synthesizer Agent': '#00B88D', // Aurora Teal 700
};
