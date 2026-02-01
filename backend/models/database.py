# SQLAlchemy models for Users, ResearchProjects, AgentPlans, PaperReferences
from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    institution = Column(String)
    tier = Column(String, default="free")  # free, pro, enterprise
    monthly_budget_usd = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    research_projects = relationship("ResearchProject", back_populates="user")
    usage_records = relationship("UserUsage", back_populates="user")
    llm_interactions = relationship("LLMInteraction", back_populates="user")


class ResearchProject(Base):
    __tablename__ = "research_projects"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    research_question = Column(Text, nullable=False)
    keywords = Column(JSON)
    subtopics = Column(JSON)
    status = Column(String, default="planning")
    total_papers_found = Column(Integer, default=0)  # <-- ADDED
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="research_projects")
    agent_plans = relationship("AgentPlan", back_populates="project")
    paper_references = relationship("PaperReference", back_populates="project")


# AgentPlan model for plan management
class AgentPlan(Base):
    __tablename__ = "agent_plans"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String, ForeignKey("research_projects.id"), nullable=False)
    agent_type = Column(String)  # planner, analyzer, synthesizer
    plan_steps = Column(JSON)  # [{step, status, output}]
    current_step = Column(Integer, default=0)
    plan_metadata = Column(JSON)
    project = relationship("ResearchProject", back_populates="agent_plans")


# PaperReference model for results
class PaperReference(Base):
    __tablename__ = "paper_references"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String, ForeignKey("research_projects.id"), nullable=False)
    title = Column(String)
    authors = Column(JSON)
    abstract = Column(Text)
    url = Column(String)
    embeddings = Column(JSON)  # vector[1536]
    relevance_score = Column(Float)
    project = relationship("ResearchProject", back_populates="paper_references")


# UserUsage model for tracking monthly usage
class UserUsage(Base):
    __tablename__ = "user_usage"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    month = Column(Date, nullable=False)  # First day of the month
    total_tokens = Column(Integer, default=0)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_cost_usd = Column(Float, default=0.0)
    projects_created = Column(Integer, default=0)
    papers_analyzed = Column(Integer, default=0)
    llm_calls = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="usage_records")


# LLMInteraction model for detailed LLM call logging
class LLMInteraction(Base):
    __tablename__ = "llm_interactions"
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    project_id = Column(String, ForeignKey("research_projects.id"), nullable=True)
    agent_type = Column(String)  # planner, analyzer, synthesizer, etc.
    model = Column(String)  # gemini-2.0-flash-lite, gemini-2.0-flash, etc.
    task_type = Column(String)  # paper_analysis, synthesis, planning, etc.
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    latency_ms = Column(Integer, default=0)
    prompt_preview = Column(Text)  # First N chars of prompt
    response_preview = Column(Text)  # First N chars of response
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="llm_interactions")
