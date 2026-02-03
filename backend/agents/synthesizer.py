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

# Maximum characters to send in a single synthesis request
# IMPORTANT: Groq free tier has 6K TPM limit. We use 16K chars (~4K tokens)
# to stay safely under that limit with room for response tokens.
# See: https://console.groq.com/docs/rate-limits
MAX_SYNTHESIS_CHARS = 16000

# Maximum tokens for a single request (conservative estimate)
# Groq TPM limits: 6K (llama-8B), 12K (llama-70B), 6K (qwen-32B)
MAX_REQUEST_TOKENS = 4000


class SynthesisExecutorAgent:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return len(text) // 4

    def synthesize_section(self, subtopic, paper_analyses, academic_level, word_count):
        """
        Synthesize a literature review section from analyzed papers.

        Handles large inputs by:
        1. ALWAYS validate token count BEFORE sending to avoid 413 errors
        2. If paper_analyses fits in one call (~4K tokens), do single synthesis
        3. If too large, split into chunks, synthesize each, then combine
        4. If a single chunk still fails, summarize it first

        This ensures we NEVER fail the project due to payload size (413 errors).
        """
        # Calculate total length and estimated tokens
        total_length = len(paper_analyses)
        estimated_tokens = self._estimate_tokens(paper_analyses)

        logger.info(
            f"Synthesis input: {total_length} chars (~{estimated_tokens} tokens). "
            f"Max allowed: {MAX_SYNTHESIS_CHARS} chars (~{MAX_REQUEST_TOKENS} tokens)"
        )

        # Check both character count AND token estimate
        if total_length <= MAX_SYNTHESIS_CHARS and estimated_tokens <= MAX_REQUEST_TOKENS:
            # Small enough for single synthesis
            logger.info("Input size OK for single synthesis")
            return self._single_synthesis(subtopic, paper_analyses, academic_level, word_count)
        else:
            # Need to chunk and synthesize in multiple passes
            logger.info(
                f"Paper analyses too large ({total_length} chars / ~{estimated_tokens} tokens), "
                f"using chunked synthesis approach (limit: {MAX_REQUEST_TOKENS} tokens/request)"
            )
            return self._chunked_synthesis(subtopic, paper_analyses, academic_level, word_count)

    def _single_synthesis(self, subtopic, paper_analyses, academic_level, word_count):
        """
        Perform single-pass synthesis for smaller inputs.

        If payload is still too large (413 error), iteratively reduce until it works.
        """
        # Build prompt and validate token count BEFORE sending
        base_prompt = f"""You are a Synthesis Executor agent specializing in academic writing and literature review generation.
        TASK: Create a literature review section synthesizing the following analyzed papers.
        Section Topic: {subtopic}
        Writing Style: Academic, formal, suitable for {academic_level} level
        Target Length: {word_count} words
        OUTPUT FORMAT:
        1. Section Introduction: Overview of topic and scope
        2. Thematic Organization: Group findings by themes/approaches
        3. Critical Analysis: Compare/contrast findings, identify patterns
        4. Research Gaps: Highlight limitations and future directions
        5. Section Conclusion: Synthesize key takeaways
        6. DON'T include References section in the end

        Analyzed Papers:
        """

        base_token_estimate = self._estimate_tokens(base_prompt)
        available_tokens = (
            MAX_REQUEST_TOKENS - base_token_estimate - 500
        )  # Reserve 500 for response
        available_chars = available_tokens * 4

        # Truncate paper_analyses if needed BEFORE building the prompt
        current_analyses = paper_analyses
        if len(current_analyses) > available_chars:
            logger.warning(
                f"Pre-truncating paper analyses from {len(current_analyses)} "
                f"to {available_chars} chars to fit token limit"
            )
            current_analyses = (
                current_analyses[:available_chars] + "\n\n[Content truncated to fit token limits]"
            )

        prompt = base_prompt + current_analyses

        # Attempt with retry and progressive reduction
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(
                    f"Synthesis attempt {attempt + 1}/{max_attempts}: "
                    f"~{self._estimate_tokens(prompt)} tokens"
                )
                return self.llm_client.chat(prompt)
            except Exception as e:
                error_str = str(e).lower()
                if "413" in str(e) or "payload" in error_str or "too large" in error_str:
                    if attempt < max_attempts - 1:
                        # Reduce prompt size by 40% and retry
                        logger.warning(
                            f"Synthesis attempt {attempt + 1} failed (413), "
                            f"reducing by 40% and retrying"
                        )
                        # Reduce the analyses portion
                        current_analyses = current_analyses[: int(len(current_analyses) * 0.6)]
                        current_analyses += "\n\n[Content truncated to fit token limits]"
                        prompt = base_prompt + current_analyses
                    else:
                        # Last attempt - use summarized approach
                        logger.warning(
                            "All synthesis attempts failed, falling back to summary approach"
                        )
                        return self._synthesize_with_summary(
                            subtopic, paper_analyses, academic_level, word_count
                        )
                else:
                    raise

        # Should not reach here, but just in case
        return self._synthesize_with_summary(subtopic, paper_analyses, academic_level, word_count)

    def _chunked_synthesis(self, subtopic, paper_analyses, academic_level, word_count):
        """
        Perform multi-pass synthesis for large inputs.

        Strategy:
        1. Split papers into chunks (by the --- separator)
        2. Summarize each chunk to extract key findings (~500 words each)
        3. Combine summaries and synthesize into final output

        IMPORTANT: Each chunk must stay under MAX_REQUEST_TOKENS to avoid 413 errors.
        """
        # Split by paper separator
        papers = paper_analyses.split("\n\n---\n\n")

        if len(papers) <= 1:
            # No clear separator, try by newlines
            papers = [p.strip() for p in paper_analyses.split("\n\n\n") if p.strip()]

        logger.info(f"Chunked synthesis: splitting {len(papers)} papers into chunks")

        # Use a smaller chunk size to ensure we stay under TPM limits
        # Reserve tokens for prompt template (~200) and response (~500)
        chunk_max_chars = (MAX_REQUEST_TOKENS - 700) * 4  # ~3300 tokens max for content

        # Group papers into chunks that fit within limits
        chunks = []
        current_chunk = []
        current_length = 0

        for paper in papers:
            paper_length = len(paper)

            # If a single paper is too large, we'll truncate it later
            if paper_length > chunk_max_chars:
                # Finish current chunk first
                if current_chunk:
                    chunks.append("\n\n---\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                # Add truncated version of large paper as its own chunk
                truncated_paper = paper[:chunk_max_chars] + "\n[Truncated]"
                chunks.append(truncated_paper)
            elif current_length + paper_length > chunk_max_chars and current_chunk:
                # Current chunk is full, start a new one
                chunks.append("\n\n---\n\n".join(current_chunk))
                current_chunk = [paper]
                current_length = paper_length
            else:
                # Add to current chunk
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
        """
        Summarize a chunk of paper analyses into key findings.

        This method ensures the chunk fits within token limits before sending.
        """
        base_prompt = f"""Summarize the key findings from these paper analyses related to "{subtopic}".
        Focus on: main contributions, methodologies, key results, and limitations.
        Keep the summary concise but comprehensive (~500 words).

        Provide a structured summary with the main themes and findings.

        Paper Analyses:
        """

        # Calculate available space for chunk content
        base_tokens = self._estimate_tokens(base_prompt)
        available_tokens = MAX_REQUEST_TOKENS - base_tokens - 600  # Reserve for response
        available_chars = available_tokens * 4

        # Truncate chunk if needed
        current_chunk = chunk
        if len(current_chunk) > available_chars:
            logger.warning(
                f"Chunk too large ({len(current_chunk)} chars), "
                f"truncating to {available_chars} chars"
            )
            current_chunk = current_chunk[:available_chars] + "\n\n[Content truncated...]"

        prompt = base_prompt + current_chunk

        try:
            return self.llm_client.chat(prompt)
        except Exception as e:
            error_str = str(e).lower()
            if "413" in str(e) or "payload" in error_str or "too large" in error_str:
                # Still too large - reduce by half and retry
                logger.warning("Chunk summarization failed with 413, reducing further")
                reduced_chunk = current_chunk[: len(current_chunk) // 2]
                reduced_chunk += "\n\n[Content truncated to fit limits...]"
                reduced_prompt = base_prompt + reduced_chunk
                return self.llm_client.chat(reduced_prompt)
            raise

    def _synthesize_with_summary(self, subtopic, paper_analyses, academic_level, word_count):
        """
        Fallback: summarize first, then synthesize.

        This is the last-resort method when all other synthesis approaches fail.
        It guarantees completion by using aggressive truncation.
        """
        # Calculate maximum content we can include in summary request
        summary_template_tokens = 200  # Estimated
        available_for_content = (MAX_REQUEST_TOKENS - summary_template_tokens - 500) * 4

        # Truncate paper_analyses to fit
        truncated_analyses = paper_analyses[:available_for_content]
        if len(truncated_analyses) < len(paper_analyses):
            truncated_analyses += "\n\n[Content truncated to fit token limits]"

        # First, create a summary of all papers
        summary_prompt = f"""Create a comprehensive summary of these research paper analyses on "{subtopic}".
        Extract the main themes, key findings, methodologies used, and research gaps.
        Keep summary under 1500 words.

        Paper Analyses:
        {truncated_analyses}"""

        logger.info(f"Summary fallback: using ~{self._estimate_tokens(summary_prompt)} tokens")

        try:
            summary = self.llm_client.chat(summary_prompt)
        except Exception as e:
            # Even summary failed - use ultra-minimal approach
            logger.error(f"Summary request failed: {e}. Using minimal summary.")
            # Take just first 2000 chars and create basic summary
            minimal_content = paper_analyses[:2000]
            summary = f"""Based on limited available data:

            Key Points from Research:
            {minimal_content[:1000]}

            [Note: Full analysis unavailable due to processing limits]"""

        # Now synthesize from the summary
        synthesis_prompt = f"""You are a Synthesis Executor agent specializing in academic writing.
        TASK: Create a literature review section based on the following research summary.
        Section Topic: {subtopic}
        Writing Style: Academic, formal, suitable for {academic_level} level
        Target Length: {word_count} words
        OUTPUT FORMAT:
        1. Section Introduction: Overview of topic and scope
        2. Thematic Organization: Group findings by themes/approaches
        3. Critical Analysis: Compare/contrast findings, identify patterns
        4. Research Gaps: Highlight limitations and future directions
        5. Section Conclusion: Synthesize key takeaways
        6. DON'T include References section in the end

        Research Summary:
        {summary}"""

        logger.info(f"Final synthesis: using ~{self._estimate_tokens(synthesis_prompt)} tokens")

        try:
            return self.llm_client.chat(synthesis_prompt)
        except Exception as e:
            # Complete failure - return the summary as the final output
            logger.error(f"Final synthesis failed: {e}. Returning summary as output.")
            return f"""Literature Review: {subtopic}

{summary}

[Note: This is a summarized review. Full synthesis unavailable due to processing constraints.]"""
