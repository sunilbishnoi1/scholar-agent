class PaperAnalyzerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def analyze_paper(self, title, abstract, content, research_question):
        prompt = f"""You are a Paper Analyzer agent specializing in academic content extraction.
        TASK: Analyze the academic paper for its relevance to the research context and extract key insights.
        
        Paper Title: "{title}"
        Abstract: "{abstract}"
        Research Context: "{research_question}"
        
        OUTPUT: Provide your analysis in a structured JSON format. The JSON object must contain the following keys:
        - "relevance_score": An integer between 0 and 100.
        - "justification": A brief (1-2 sentence) justification for the score.
        - "key_findings": A list of 3-5 strings, where each string is a primary research finding.
        - "methodology": A brief description of the research methods and sample.
        - "limitations": A list of strings, where each string is an identified study limitation.
        - "contribution": A brief description of how this paper fits into the broader research landscape.
        - "key_quotes": A list of 2-3 strings of key quotable insights.

        Example JSON output format:
        {{
            "relevance_score": 85,
            "justification": "The paper directly addresses the use of machine learning to predict student performance, which is highly relevant to the research question.",
            "key_findings": ["Finding 1...", "Finding 2..."],
            "methodology": "The study used a quantitative approach with a sample of 500 university students.",
            "limitations": ["The study was conducted at a single institution.", "The dataset used was limited in scope."],
            "contribution": "This work provides a foundational model for predicting student success factors.",
            "key_quotes": ["Quote 1...", "Quote 2..."]
        }}
        
        CRITICAL: Output ONLY the JSON object. Do NOT include any preamble text (like "Here's the analysis" or "Here is the JSON"), explanations, or markdown code blocks. Start your response directly with {{ and end with }}.
        """
        return self.llm_client.chat(prompt)
