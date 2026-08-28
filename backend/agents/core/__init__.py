"""
Scholar Agent Core Reasoning Specialist Agents.
"""

from .auditor import AuditorAgent, DeterministicCitationAuditorAgent
from .critic import AdversarialCriticAgent, CriticAgent
from .discovery import AutonomousLiteratureExplorer, DiscoveryAgent
from .ingestion import FullTextIngestionSpecialist, IngestionAgent
from .matrix_builder import EvidenceMatrixBuilder, MatrixBuilderAgent
from .supervisor import (
    AutonomousSupervisorAgent,
    build_scholar_agent_graph,
    finalizer_node,
    should_refine_or_finalize,
)
from .synthesizer import (
    SectionAwareContextPacker,
    SynthesizerAgent,
    ThematicSynthesizerAgent,
)


__all__ = [
    "AutonomousLiteratureExplorer",
    "DiscoveryAgent",
    "FullTextIngestionSpecialist",
    "IngestionAgent",
    "EvidenceMatrixBuilder",
    "MatrixBuilderAgent",
    "ThematicSynthesizerAgent",
    "SynthesizerAgent",
    "SectionAwareContextPacker",
    "AdversarialCriticAgent",
    "CriticAgent",
    "DeterministicCitationAuditorAgent",
    "AuditorAgent",
    "AutonomousSupervisorAgent",
    "build_scholar_agent_graph",
    "should_refine_or_finalize",
    "finalizer_node",
]

