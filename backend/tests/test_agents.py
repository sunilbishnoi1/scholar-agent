import pytest

from agents.analyzer import PaperAnalyzerAgent
from agents.planner import ResearchPlannerAgent
from agents.synthesizer import SynthesisExecutorAgent


def test_research_planner_stub():
    agent = ResearchPlannerAgent(llm_client=None)
    assert hasattr(agent, "generate_initial_plan")


def test_paper_analyzer_stub():
    agent = PaperAnalyzerAgent(llm_client=None)
    assert hasattr(agent, "analyze_paper")


def test_synthesizer_stub():
    agent = SynthesisExecutorAgent(llm_client=None)
    assert hasattr(agent, "synthesize_section")
