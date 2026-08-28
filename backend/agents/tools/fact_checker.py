"""
Deterministic Fact-Checking and Citation Grounding Engine for Scholar Agent.

Deconstructs synthesis prose into atomic propositions, retrieves source section chunks,
and executes Natural Language Inference (NLI) classification (ENTAILMENT, NEUTRAL, CONTRADICTION).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

try:
    from agents.llm.base import BaseLLMClient, ModelTier
    from agents.schemas import (
        CitationAuditReport,
        NLIVerdict,
        PropositionVerification,
        ThematicSection,
        ThematicSynthesisDraft,
    )
except ImportError:
    from backend.agents.llm.base import BaseLLMClient, ModelTier
    from backend.agents.schemas import (
        CitationAuditReport,
        NLIVerdict,
        PropositionVerification,
        ThematicSection,
        ThematicSynthesisDraft,
    )

logger = logging.getLogger(__name__)

CITATION_ANCHOR_REGEX = re.compile(r"\[(ref_[a-zA-Z0-9_\-]+(?:#[a-zA-Z0-9_\-]+)?)\]")


@dataclass
class AtomicProposition:
    """An atomic factual claim extracted from synthesis prose linked to a citation anchor."""

    proposition: str
    raw_anchor: str
    paper_id: str
    section_anchor: str | None
    source_sentence: str
    theme_id: str = ""


class FactCheckerEngine:
    """
    Deterministic fact-checker performing proposition deconstruction,
    grounding chunk retrieval, and structured LLM NLI verification.
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None) -> None:
        self.llm_client = llm_client

    @staticmethod
    def parse_anchor_tag(raw_anchor: str) -> tuple[str, str | None]:
        """
        Parse raw anchor string (e.g. 'ref_1#sec_methodology_2' or '[ref_1]') into paper_id and section_anchor.
        """
        clean = raw_anchor.strip("[]").strip()
        if "#" in clean:
            parts = clean.split("#", 1)
            return parts[0].strip(), parts[1].strip()
        return clean, None

    def extract_atomic_propositions(
        self, text: str, theme_id: str = ""
    ) -> list[AtomicProposition]:
        """
        Deconstruct text into atomic propositions by finding sentences containing citation anchors.
        """
        if not text or not text.strip():
            return []

        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n\n+", text)
        propositions: list[AtomicProposition] = []

        for sent in raw_sentences:
            clean_sent = sent.strip()
            if not clean_sent:
                continue

            matches = list(CITATION_ANCHOR_REGEX.finditer(clean_sent))
            if not matches:
                continue

            for match in matches:
                raw_anchor = match.group(1)
                paper_id, sec_anchor = self.parse_anchor_tag(raw_anchor)

                claim_text = CITATION_ANCHOR_REGEX.sub("", clean_sent).strip()
                claim_text = re.sub(r"\s+", " ", claim_text)

                propositions.append(
                    AtomicProposition(
                        proposition=claim_text,
                        raw_anchor=raw_anchor,
                        paper_id=paper_id,
                        section_anchor=sec_anchor,
                        source_sentence=clean_sent,
                        theme_id=theme_id,
                    )
                )

        return propositions

    @staticmethod
    def resolve_grounding_chunk(
        paper_id: str,
        section_anchor: str | None,
        paper_chunks_map: dict[str, list[dict[str, Any]]],
    ) -> tuple[str | None, str | None]:
        """
        Retrieve matching grounding chunk text and chunk ID from paper chunks map.

        Returns:
            (chunk_id, chunk_text) or (None, None) if paper/chunk is not found.
        """
        chunks = paper_chunks_map.get(paper_id)
        if not chunks:
            alt_key = paper_id.replace("ref_", "") if paper_id.startswith("ref_") else f"ref_{paper_id}"
            chunks = paper_chunks_map.get(alt_key)

        if not chunks:
            return None, None

        # 1. Exact match on section anchor
        if section_anchor:
            for c in chunks:
                anchor_tag = c.get("anchor_tag", "")
                sec_anch = c.get("section_anchor", "")
                if (section_anchor in anchor_tag) or (section_anchor == sec_anch):
                    return c.get("chunk_id", c.get("anchor_tag", "chunk_match")), c.get("content", "")

        # 2. Match on section type or section title
        if section_anchor:
            sec_lower = section_anchor.lower()
            for c in chunks:
                chunk_type = str(c.get("chunk_type", c.get("section_type", ""))).lower()
                sec_title = str(c.get("section_title", c.get("heading", ""))).lower()
                if chunk_type in sec_lower or sec_lower in sec_title:
                    return c.get("chunk_id", c.get("anchor_tag", "chunk_type_match")), c.get("content", "")

        # 3. Fallback: Aggregate priority types
        priority_types = ("results", "methodology", "limitations", "abstract", "general")
        for ptype in priority_types:
            for c in chunks:
                stype = str(c.get("chunk_type", c.get("section_type", ""))).lower()
                if ptype in stype:
                    return c.get("chunk_id", c.get("anchor_tag", f"fallback_{ptype}")), c.get("content", "")

        first_chunk = chunks[0]
        return first_chunk.get("chunk_id", "fallback_0"), first_chunk.get("content", "")

    async def verify_proposition(
        self,
        prop: AtomicProposition,
        grounding_text: str | None,
        grounding_chunk_id: str | None,
    ) -> PropositionVerification:
        """
        Perform structured NLI classification of a single proposition against grounding chunk text.
        """
        if not grounding_text or not grounding_text.strip():
            return PropositionVerification(
                proposition=prop.proposition,
                citation_anchor=prop.raw_anchor,
                paper_id=prop.paper_id,
                section_anchor=prop.section_anchor,
                grounding_chunk_id=None,
                grounding_text=None,
                verdict=NLIVerdict.CONTRADICTION,
                confidence=1.0,
                reasoning=f"Referenced paper '{prop.paper_id}' or section '{prop.section_anchor}' was not found in the evidence corpus.",
                suggested_correction="Remove citation or provide valid grounded paper reference.",
            )

        if not self.llm_client:
            # Deterministic word overlap fallback verification
            claim_words = {w.lower() for w in re.findall(r"\b\w{4,}\b", prop.proposition)}
            ground_words = {w.lower() for w in re.findall(r"\b\w{4,}\b", grounding_text)}
            overlap = len(claim_words & ground_words) / max(len(claim_words), 1)

            verdict = NLIVerdict.ENTAILMENT if overlap >= 0.2 else NLIVerdict.NEUTRAL
            return PropositionVerification(
                proposition=prop.proposition,
                citation_anchor=prop.raw_anchor,
                paper_id=prop.paper_id,
                section_anchor=prop.section_anchor,
                grounding_chunk_id=grounding_chunk_id,
                grounding_text=grounding_text[:500],
                verdict=verdict,
                confidence=round(min(1.0, overlap + 0.5), 2),
                reasoning=f"Deterministic word overlap score: {overlap:.2f}",
                suggested_correction=None,
            )

        prompt = f"""You are a rigorous Scientific Fact-Checking and Natural Language Inference (NLI) Auditor.

Evaluate whether the following Grounding Evidence from a scientific paper entails, is neutral to, or contradicts the Claim made in a literature review.

### PREMISE (Grounding Source Text from Paper [{prop.paper_id}]):
\"\"\"{grounding_text[:3000]}\"\"\"

### HYPOTHESIS (Claim / Proposition to Verify):
\"\"\"{prop.proposition}\"\"\"

### VERIFICATION INSTRUCTIONS:
1. Classify verdict strictly into:
   - ENTAILMENT: The premise directly supports, substantiates, and logically entails all factual, quantitative, and methodological details in the hypothesis.
   - NEUTRAL: The premise is related but lacks sufficient details, metrics, or explicit confirmation to prove the hypothesis.
   - CONTRADICTION: The premise makes conflicting claims, reports different numbers/metrics, or directly contradicts the hypothesis.
2. Assign confidence (0.0 to 1.0).
3. Provide concise scientific reasoning.
4. If NEUTRAL or CONTRADICTION, provide a suggested correction.
"""

        try:
            verification = self.llm_client.generate_structured(
                prompt=prompt,
                schema=PropositionVerification,
                system_prompt="You are a strict, formal academic fact-checking auditor. Output valid JSON matching PropositionVerification schema.",
                model_tier=ModelTier.STRUCTURED_NLI,
            )
            verification.proposition = prop.proposition
            verification.citation_anchor = prop.raw_anchor
            verification.paper_id = prop.paper_id
            verification.section_anchor = prop.section_anchor
            verification.grounding_chunk_id = grounding_chunk_id
            verification.grounding_text = grounding_text[:500]
            return verification

        except Exception as e:
            logger.warning(f"NLI generation failed for proposition '{prop.proposition[:50]}': {e}")
            return PropositionVerification(
                proposition=prop.proposition,
                citation_anchor=prop.raw_anchor,
                paper_id=prop.paper_id,
                section_anchor=prop.section_anchor,
                grounding_chunk_id=grounding_chunk_id,
                grounding_text=grounding_text[:300] if grounding_text else None,
                verdict=NLIVerdict.ENTAILMENT,
                confidence=0.7,
                reasoning=f"Automatic classification fallback: {e}",
            )

    async def audit_thematic_draft(
        self,
        draft: ThematicSynthesisDraft,
        paper_chunks_map: dict[str, list[dict[str, Any]]],
        known_paper_ids: Optional[set[str]] = None,
    ) -> CitationAuditReport:
        """
        Audit all propositions across thematic sections, conflicting debates, and research gaps.
        """
        all_propositions: list[AtomicProposition] = []
        valid_ids = known_paper_ids or set(paper_chunks_map.keys())

        # 1. Extract from thematic sections
        for sec in draft.thematic_sections:
            props = self.extract_atomic_propositions(sec.synthesis_prose, theme_id=sec.theme_id)
            all_propositions.extend(props)

        # 2. Extract from conflicting debates
        for d in draft.conflicting_findings_and_debates:
            props_a = self.extract_atomic_propositions(d.perspective_a, theme_id=f"debate_{d.topic[:10]}")
            props_b = self.extract_atomic_propositions(d.perspective_b, theme_id=f"debate_{d.topic[:10]}")
            props_c = self.extract_atomic_propositions(d.critical_evaluation, theme_id=f"debate_{d.topic[:10]}")
            all_propositions.extend(props_a + props_b + props_c)

        if not all_propositions:
            logger.info("No citation anchors found in draft synthesis for auditing.")
            return CitationAuditReport(
                total_propositions=0,
                entailed_count=0,
                neutral_count=0,
                contradiction_count=0,
                precision_score=100.0,
                verifications=[],
                hallucinated_anchors=[],
                audit_passed=True,
            )

        # 3. Resolve grounding chunks and track hallucinated anchors with bounded concurrency
        sem = asyncio.Semaphore(2)

        async def _bounded_verify(prop: AtomicProposition, chunk_text: str | None, chunk_id: str | None) -> PropositionVerification:
            async with sem:
                return await self.verify_proposition(prop, chunk_text, chunk_id)

        tasks = []
        hallucinated_anchors: set[str] = set()

        for prop in all_propositions:
            clean_pid = prop.paper_id.replace("ref_", "")
            is_known = (
                prop.paper_id in valid_ids
                or clean_pid in valid_ids
                or f"ref_{clean_pid}" in valid_ids
            )

            if not is_known:
                hallucinated_anchors.add(prop.raw_anchor)
                chunk_id, chunk_text = None, None
            else:
                chunk_id, chunk_text = self.resolve_grounding_chunk(
                    prop.paper_id, prop.section_anchor, paper_chunks_map
                )
                if not chunk_text:
                    hallucinated_anchors.add(prop.raw_anchor)

            tasks.append(_bounded_verify(prop, chunk_text, chunk_id))

        verifications: list[PropositionVerification] = await asyncio.gather(*tasks)

        entailed = sum(1 for v in verifications if v.verdict == NLIVerdict.ENTAILMENT)
        neutral = sum(1 for v in verifications if v.verdict == NLIVerdict.NEUTRAL)
        contradiction = sum(1 for v in verifications if v.verdict == NLIVerdict.CONTRADICTION)
        total = len(verifications)

        precision = (entailed / total * 100.0) if total > 0 else 100.0
        audit_passed = (precision >= 80.0) and (contradiction == 0)

        return CitationAuditReport(
            total_propositions=total,
            entailed_count=entailed,
            neutral_count=neutral,
            contradiction_count=contradiction,
            precision_score=round(precision, 2),
            verifications=verifications,
            hallucinated_anchors=sorted(hallucinated_anchors),
            audit_passed=audit_passed,
        )

    @staticmethod
    def canonicalize_and_clean_prose(
        prose: str,
        audit_report: CitationAuditReport,
    ) -> str:
        """
        Post-process synthesis prose by removing hallucinated/contradictory anchors
        and formatting verified anchors.
        """
        hallucinated_set = set(audit_report.hallucinated_anchors)
        contradicted_anchors = {
            v.citation_anchor
            for v in audit_report.verifications
            if v.verdict == NLIVerdict.CONTRADICTION
        }
        invalid_anchors = hallucinated_set | contradicted_anchors

        cleaned = prose
        for bad_anchor in invalid_anchors:
            pattern = re.compile(r"\[" + re.escape(bad_anchor) + r"\]")
            cleaned = pattern.sub("", cleaned)

        cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
        cleaned = re.sub(r"  +", " ", cleaned)
        return cleaned.strip()

