# Synthesis Executor Agent
# Role: Combines analyzed papers into coherent literature review sections,
# maintains academic writing style, ensures proper citations, identifying research gaps.
#
# IMPORTANT: This agent handles large inputs gracefully by:
# 1. Chunking paper analyses if too large for a single LLM call
# 2. Synthesizing in multiple passes if needed
# 3. NEVER failing the project due to payload size

import logging

logger = logging.getLogger(__name__)

# Maximum characters to send in a single synthesis request (~10k tokens)
MAX_SYNTHESIS_CHARS = 40000


class SynthesisExecutorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def synthesize_section(self, subtopic, paper_analyses, academic_level, word_count):
        """
        Synthesize a literature review section from analyzed papers.

        Handles large inputs by:
        1. If paper_analyses fits in one call, do single synthesis
        2. If too large, split into chunks, synthesize each, then combine
        3. If a single chunk still fails, summarize it first

        This ensures we NEVER fail the project due to payload size.
        """
        # Estimate if we need to chunk
        total_length = len(paper_analyses)

        if total_length <= MAX_SYNTHESIS_CHARS:
            # Small enough for single synthesis
            return self._single_synthesis(subtopic, paper_analyses, academic_level, word_count)
        else:
            # Need to chunk and synthesize in multiple passes
            logger.info(
                f"Paper analyses too large ({total_length} chars), "
                f"using chunked synthesis approach"
            )
            return self._chunked_synthesis(subtopic, paper_analyses, academic_level, word_count)

    def _single_synthesis(self, subtopic, paper_analyses, academic_level, word_count):
        """Perform single-pass synthesis for smaller inputs."""
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

        try:
            return self.llm_client.chat(prompt)
        except Exception as e:
            # If synthesis fails (e.g., still too large), try with summarized content
            if "413" in str(e) or "payload" in str(e).lower():
                logger.warning(
                    "Single synthesis failed with payload error, trying summarized approach"
                )
                return self._synthesize_with_summary(
                    subtopic, paper_analyses, academic_level, word_count
                )
            raise

    def _chunked_synthesis(self, subtopic, paper_analyses, academic_level, word_count):
        """
        Perform multi-pass synthesis for large inputs.

        Strategy:
        1. Split papers into chunks (by the --- separator)
        2. Summarize each chunk to extract key findings
        3. Synthesize the summaries into final output
        """
        # Split by paper separator
        papers = paper_analyses.split("\n\n---\n\n")

        if len(papers) <= 1:
            # No clear separator, try by newlines
            papers = [p.strip() for p in paper_analyses.split("\n\n\n") if p.strip()]

        # Group papers into chunks that fit within limits
        chunks = []
        current_chunk = []
        current_length = 0

        for paper in papers:
            paper_length = len(paper)
            if current_length + paper_length > MAX_SYNTHESIS_CHARS and current_chunk:
                chunks.append("\n\n---\n\n".join(current_chunk))
                current_chunk = [paper]
                current_length = paper_length
            else:
                current_chunk.append(paper)
                current_length += paper_length

        if current_chunk:
            chunks.append("\n\n---\n\n".join(current_chunk))

        logger.info(f"Split into {len(chunks)} chunks for synthesis")

        # Synthesize each chunk into key findings
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            summary = self._summarize_chunk(chunk, subtopic)
            chunk_summaries.append(summary)

        # Final synthesis from summaries
        combined_summaries = "\n\n---\n\n".join(chunk_summaries)

        if len(combined_summaries) > MAX_SYNTHESIS_CHARS:
            # Still too large, do recursive summarization
            logger.warning("Combined summaries still large, doing recursive summarization")
            return self._chunked_synthesis(subtopic, combined_summaries, academic_level, word_count)

        # Final synthesis
        return self._single_synthesis(subtopic, combined_summaries, academic_level, word_count)

    def _summarize_chunk(self, chunk, subtopic):
        """Summarize a chunk of paper analyses into key findings."""
        prompt = f"""Summarize the key findings from these paper analyses related to "{subtopic}".
        Focus on: main contributions, methodologies, key results, and limitations.
        Keep the summary concise but comprehensive (~500 words).

        Paper Analyses:
        {chunk}

        Provide a structured summary with the main themes and findings."""

        try:
            return self.llm_client.chat(prompt)
        except Exception as e:
            if "413" in str(e) or "payload" in str(e).lower():
                # Chunk still too large, truncate and retry
                logger.warning("Chunk too large for summarization, truncating")
                truncated = chunk[: MAX_SYNTHESIS_CHARS // 2] + "\n\n[Content truncated...]"
                return self.llm_client.chat(
                    f"""Summarize the key findings from these paper analyses related to "{subtopic}".
                    Paper Analyses (truncated):
                    {truncated}

                    Provide a brief summary of the main themes and findings."""
                )
            raise

    def _synthesize_with_summary(self, subtopic, paper_analyses, academic_level, word_count):
        """Fallback: summarize first, then synthesize."""
        # First, create a summary of all papers
        summary_prompt = f"""Create a comprehensive summary of these research paper analyses on "{subtopic}".
        Extract the main themes, key findings, methodologies used, and research gaps.
        Keep summary under 2000 words.

        Paper Analyses:
        {paper_analyses[:MAX_SYNTHESIS_CHARS]}

        [Note: Some content may have been truncated due to length]"""

        summary = self.llm_client.chat(summary_prompt)

        # Now synthesize from the summary
        synthesis_prompt = f"""You are a Synthesis Executor agent specializing in academic writing.
        TASK: Create a literature review section based on the following research summary.
        Section Topic: {subtopic}
        Research Summary: {summary}
        Writing Style: Academic, formal, suitable for {academic_level} level
        Target Length: {word_count} words
        OUTPUT FORMAT:
        1. Section Introduction: Overview of topic and scope
        2. Thematic Organization: Group findings by themes/approaches
        3. Critical Analysis: Compare/contrast findings, identify patterns
        4. Research Gaps: Highlight limitations and future directions
        5. Section Conclusion: Synthesize key takeaways
        6. DON'T include References section in the end"""

        return self.llm_client.chat(synthesis_prompt)
