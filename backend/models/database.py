# SQLAlchemy models for Users, ResearchProjects, AgentPlans, PaperReferences
from sqlalchemy import Column, String, DateTime, Text, Integer, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False) 
    institution = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    research_projects = relationship('ResearchProject', back_populates='user')


class ResearchProject(Base):
    __tablename__ = 'research_projects'
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    title = Column(String, nullable=False)
    research_question = Column(Text, nullable=False)
    keywords = Column(JSON)
    subtopics = Column(JSON)
    status = Column(String, default='planning')
    total_papers_found = Column(Integer, default=0) # <-- ADDED
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship('User', back_populates='research_projects')
    agent_plans = relationship('AgentPlan', back_populates='project')
    paper_references = relationship('PaperReference', back_populates='project')


# AgentPlan model for plan management
class AgentPlan(Base):
    __tablename__ = 'agent_plans'
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String, ForeignKey('research_projects.id'), nullable=False)
    agent_type = Column(String)  # planner, analyzer, synthesizer
    plan_steps = Column(JSON)    # [{step, status, output}]
    current_step = Column(Integer, default=0)
    plan_metadata = Column(JSON)
    project = relationship('ResearchProject', back_populates='agent_plans')

# PaperReference model for results
class PaperReference(Base):
    __tablename__ = 'paper_references'
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String, ForeignKey('research_projects.id'), nullable=False)
    title = Column(String)
    authors = Column(JSON)
    abstract = Column(Text)
    url = Column(String)
    embeddings = Column(JSON)  # vector[1536]
    relevance_score = Column(Float)
    project = relationship('ResearchProject', back_populates='paper_references')