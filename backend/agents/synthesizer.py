# Synthesis Executor Agent
# Role: Combines analyzed papers into coherent literature review sections, maintains academic writing style, ensures proper citations, identifying research gaps.

class SynthesisExecutorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def synthesize_section(self, subtopic, paper_analyses, academic_level, word_count):
        prompt = f"""You are a Synthesis Executor agent specializing in academic writing and literature review generation.
        TASK: Create a literature review section synthesizing the following analyzed papers.
        Section Topic: {subtopic}
        Analyzed Papers: {paper_analyses}
        Writing Style: Academic, formal, suitable for {academic_level} level
        Target Length: {word_count} words
        OUTPUT FORMAT:
        1. Section Introduction: Overview of topic and scope
        2. Thematic Organization: Group findings by themes/approaches
        3. Critical Analysis: Compare/contrast findings, identify patterns
        4. Research Gaps: Highlight limitations and future directions
        5. Section Conclusion: Synthesize key takeaways
        6. DON'T include References section in the end"""
        return self.llm_client.chat(prompt)
