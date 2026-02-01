# Research Planner Agent
# Role: Analyzes research questions, creates search strategies, identifies key subtopics and relevant academic domains inorder to finally in the end achieve the identification of research gaps.
import json
import logging
import re


class ResearchPlannerAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_initial_plan(self, research_question, title):
        """
        Generates a list of keywords and subtopics from a research question and title.
        """
        prompt = f"""
        You are a research assistant AI. Your task is to generate a list of relevant keywords
        and a list of research subtopics for a given research question and title.

        Research Question: "{research_question}"
        Title: "{title}"

        Please generate:
        1. A list of 8-12 diverse and specific keywords for searching academic databases.
        2. A list of 4-6 specific subtopics to structure the literature review.

        Provide the output as a single JSON object with two keys: "keywords" and "subtopics".
        Ensure the output is ONLY the JSON object, without any surrounding text or markdown.

        Example Output:
        {{
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "subtopics": ["Subtopic A", "Subtopic B", "Subtopic C"]
        }}
        """
        response_str = self.llm_client.chat(prompt)
        try:
            # Clean up the response in case the LLM wraps it in markdown
            clean_response = re.sub(r'```json\s*|\s*```', '', response_str).strip()
            data = json.loads(clean_response)

            keywords = data.get("keywords", [])
            subtopics = data.get("subtopics", [])

            # Basic validation
            if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords) and \
               isinstance(subtopics, list) and all(isinstance(s, str) for s in subtopics):
                return data
            else:
                logging.warning(f"LLM returned a malformed JSON object. Fallback required. Response: {response_str}")
                return {"keywords": [], "subtopics": []}

        except (json.JSONDecodeError, AttributeError) as e:
            logging.error(f"Failed to parse JSON for initial plan, returning empty. Error: {e}. Raw response: {response_str}")
            return {"keywords": [], "subtopics": []}
