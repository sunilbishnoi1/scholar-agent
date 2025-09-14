# Research Planner Agent
# Role: Analyzes research questions, creates search strategies, identifies key subtopics and relevant academic domains inorder to finally in the end achieve the identification of research gaps.

class ResearchPlannerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def create_search_strategy(self, research_question, keywords, max_papers=50):
        prompt = f"""You are a Research Planner agent specializing in academic literature discovery.
        TASK: Create a comprehensive search strategy for the following research question.
        Research Question: {research_question}
        Keywords: {keywords}
        Target Paper Count: {max_papers}
        OUTPUT FORMAT:
        1. Search Strategy: List 5-8 specific search terms and Boolean combinations
        2. Academic Domains: Identify 3-5 relevant academic fields/conferences
        3. Time Range: Recommend optimal publication date range
        4. Quality Filters: Suggest citation thresholds and venue requirements
        5. Subtopics: Break down research question into 4-6 specific subtopics"""
        return self.llm_client.chat(prompt)