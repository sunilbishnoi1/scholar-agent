import { describe, it, expect } from 'vitest';
import {
    formatAuthors,
    slugifyTitle,
    generateExportFilename,
    generateReportMarkdown,
    generateBibTeX,
    createAcademicPrintableContainer,
} from './exportEngine';
import type { ResearchReport } from '../types';

describe('exportEngine Unit Test Suite', () => {
    describe('formatAuthors', () => {
        it('handles null, undefined, or empty author lists', () => {
            expect(formatAuthors(undefined)).toBe('Unknown Authors');
            expect(formatAuthors(null)).toBe('Unknown Authors');
            expect(formatAuthors([])).toBe('Unknown Authors');
            expect(formatAuthors(['Unknown'])).toBe('Unknown Authors');
        });

        it('formats single author', () => {
            expect(formatAuthors(['Alice Smith'])).toBe('Alice Smith');
        });

        it('formats two authors with "and"', () => {
            expect(formatAuthors(['Alice Smith', 'Bob Jones'])).toBe('Alice Smith and Bob Jones');
        });

        it('formats three or more authors with "et al."', () => {
            expect(formatAuthors(['Alice Smith', 'Bob Jones', 'Charlie Brown'])).toBe('Alice Smith et al.');
        });
    });

    describe('slugifyTitle & generateExportFilename', () => {
        it('slugifies titles with special characters and whitespace', () => {
            expect(slugifyTitle('Deep Transformers: A Survey (2026)!')).toBe('deep_transformers_a_survey_2026');
            expect(slugifyTitle('   Exploring Multi-Agent LLMs   ')).toBe('exploring_multi_agent_llms');
            expect(slugifyTitle('')).toBe('literature_review');
        });

        it('generates export filename with date and extension', () => {
            const filename = generateExportFilename('Deep Learning for NLP', 'pdf');
            expect(filename).toMatch(/^deep_learning_for_nlp_\d{4}-\d{2}-\d{2}\.pdf$/);
        });
    });

    describe('generateReportMarkdown', () => {
        const mockReport: ResearchReport = {
            metadata: {
                title: 'Attention in Transformers',
                research_question: 'How does attention scale with sequence length?',
                generated_at: '2026-08-27T12:00:00Z',
                quality_score: 92.0,
                status: 'completed',
            },
            executive_summary: 'Comprehensive review of linear and sparse attention mechanisms.',
            methodology_overview: {
                dominant_approach: 'Sparse Attention',
                trend_description: 'Rapid adoption of FlashAttention and sparse routing.',
                distribution: { 'Sparse Attention': 5, 'Dense Attention': 3 },
            },
            thematic_sections: [
                {
                    theme_id: 'sec_1',
                    title: 'Memory Complexity',
                    synthesis_prose: 'Standard attention incurs O(N^2) memory complexity.',
                    key_takeaways: ['Linear attention reduces memory to O(N).'],
                    cited_paper_ids: ['ref_1'],
                },
            ],
            comparison_matrix: [
                {
                    paper_id: 'ref_1',
                    title: 'FlashAttention: Fast and Memory-Efficient Exact Attention',
                    authors: ['Tri Dao', 'Daniel Y. Fu'],
                    year: 2022,
                    methodology: 'IO-aware exact attention tiling',
                    benchmark_dataset: 'GPT-2 / BERT',
                    primary_metric: 'Speedup: 3x-5x',
                    primary_limitation: 'Requires specialized GPU SRAM tiling kernels',
                    is_full_text: true,
                },
            ],
            conflicting_findings_and_debates: [
                {
                    topic: 'Exact vs Approximate Attention',
                    perspective_a: 'Exact IO-aware tiling maintains 100% downstream accuracy.',
                    perspective_b: 'Low-rank approximations offer faster training on long sequences.',
                    critical_evaluation: 'Exact tiling is strictly dominant on modern GPU hardware architectures.',
                },
            ],
            actionable_research_gaps: [
                {
                    gap_id: 'GAP-01',
                    description: 'Lack of standardized hardware-agnostic kernel implementations.',
                    importance: 'HIGH',
                    recommended_methodology: 'Develop Triton cross-architecture memory kernels.',
                    grounding_paper_ids: ['ref_1'],
                },
            ],
            bibliography: [
                {
                    paper_id: 'ref_1',
                    title: 'FlashAttention: Fast and Memory-Efficient Exact Attention',
                    authors: ['Tri Dao', 'Daniel Y. Fu'],
                    year: 2022,
                    venue: 'NeurIPS',
                    doi: '10.48550/arXiv.2205.14135',
                    url: 'https://arxiv.org/abs/2205.14135',
                    is_full_text_analyzed: true,
                },
            ],
        };

        it('generates complete markdown with all sections', () => {
            const md = generateReportMarkdown(mockReport);
            expect(md).toContain('# Attention in Transformers');
            expect(md).toContain('**Research Question:** How does attention scale with sequence length?');
            expect(md).toContain('## Executive Summary');
            expect(md).toContain('## Evidence Comparison Matrix');
            expect(md).toContain('FlashAttention: Fast and Memory-Efficient Exact Attention');
            expect(md).toContain('## Thematic Synthesis');
            expect(md).toContain('## Conflicting Findings & Scientific Debates');
            expect(md).toContain('## Actionable Research Gaps & Future Directions');
            expect(md).toContain('[HIGH PRIORITY] GAP-01');
            expect(md).toContain('Tri Dao and Daniel Y. Fu');
        });

        it('creates an academic printable HTML container', () => {
            const container = createAcademicPrintableContainer(mockReport);
            expect(container).toBeInstanceOf(HTMLElement);
            expect(container.id).toBe('academic-pdf-export-root');
            expect(container.innerHTML).toContain('Attention in Transformers');
            expect(container.innerHTML).toContain('Evidence Comparison Matrix');
            expect(container.innerHTML).toContain('FlashAttention');
            expect(container.innerHTML).toContain('Scientific &amp; Methodological Debates');
            expect(container.innerHTML).toContain('GAP-01');
            expect(container.innerHTML).toContain('Tri Dao and Daniel Y. Fu');
        });

        it('generates standard BibTeX entries without Unknown Authors when authors exist', () => {
            const bib = generateBibTeX(mockReport.bibliography!);
            expect(bib).toContain('@article{ref_1,');
            expect(bib).toContain('title={FlashAttention: Fast and Memory-Efficient Exact Attention}');
            expect(bib).toContain('author={Tri Dao and Daniel Y. Fu}');
            expect(bib).toContain('doi={10.48550/arXiv.2205.14135}');
        });

        it('creates printable container with all 6 academic sections and correct styling', () => {
            const container = createAcademicPrintableContainer(mockReport);
            expect(container.style.backgroundColor).toBe('rgb(255, 255, 255)');
            expect(container.querySelectorAll('table').length).toBe(1);
            expect(container.innerHTML).toContain('Evidence Comparison Matrix');
            expect(container.innerHTML).toContain('Scientific &amp; Methodological Debates');
            expect(container.innerHTML).toContain('Actionable Research Gaps &amp; Future Directions');
            expect(container.innerHTML).toContain('References &amp; Bibliography');
        });
    });
});
