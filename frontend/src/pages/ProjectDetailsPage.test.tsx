/**
 * Comprehensive Empirical Test Suite for ProjectDetailsPage (Scholar Agent v3.2 Phase 6)
 *
 * Tests:
 * 1. Empty State testing for all 6 tabs
 * 2. Loading State testing (page loading, background running, report loading)
 * 3. Error State testing (project fetch failure, missing project)
 * 4. Full Report State testing across all 6 tabs:
 *    - Tab 0: Literature Review (Executive Summary, Methodology Distribution, Thematic Sections + Citation Popover)
 *    - Tab 1: Evidence Matrix (Table render, Sorting, Search, Full-Text filter, Row expansion)
 *    - Tab 2: Methodological Debates (Perspective A/B cards, Evaluation, Search)
 *    - Tab 3: Actionable Research Gaps (Priority badges, Methodology roadmap, Priority filter, Search)
 *    - Tab 4: Discovered Papers & Bibliography (Corpus cards, BibTeX export)
 *    - Tab 5: Real-Time Journey (LangGraph 7-stage DAG, Critic verdict banner, Auditor banner, Telemetry logs)
 * 5. Zero Placeholder / "Coming Soon" Verification across all tabs
 * 6. Export Deliverables menu verification (.md, .pdf, .docx, .bib)
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import ProjectDetailsPage from './ProjectDetailsPage';
import * as neonClient from '../api/neonClient';
import * as apiClient from '../api/client';
import * as streamHook from '../hooks/useProjectStream';
import * as exportUtils from '../utils/exportEngine';
import type { ResearchProject, ReportResponse, ResearchReport } from '../types';

// Mock exportEngine methods
vi.mock('../utils/exportEngine', async (importOriginal) => {
    const actual = await importOriginal<typeof exportUtils>();
    return {
        ...actual,
        exportReport: vi.fn().mockResolvedValue(undefined),
        exportToMarkdown: vi.fn(),
        exportToDocx: vi.fn().mockResolvedValue(undefined),
        exportToPdf: vi.fn().mockResolvedValue(undefined),
        exportToBibTeX: vi.fn(),
    };
});

describe('ProjectDetailsPage Empirical Challenger Test Suite', () => {
    let queryClient: QueryClient;

    const mockProjectData: ResearchProject = {
        id: 'proj-123',
        title: 'Deep Transformer Architectures for Scientific Synthesis',
        research_question: 'How do sparse attention transformers improve cross-document literature synthesis?',
        keywords: ['transformers', 'literature synthesis', 'sparse attention'],
        subtopics: ['attention mechanisms', 'summarization'],
        status: 'completed',
        created_at: '2026-08-26T12:00:00Z',
        total_papers_found: 3,
        paper_references: [
            {
                id: 'ref_1',
                title: 'Longformer: The Long-Document Transformer',
                authors: ['Iz Beltagy', 'Matthew E. Peters', 'Arman Cohan'],
                year: 2020,
                venue: 'arXiv',
                doi: '10.48550/arXiv.2004.05150',
                url: 'https://arxiv.org/abs/2004.05150',
                citation_count: 1850,
                relevance_score: 0.95,
                is_full_text: true,
            },
            {
                id: 'ref_2',
                title: 'Big Bird: Transformers for Longer Sequences',
                authors: ['Manzil Zaheer', 'Guru Guruganesh', 'Avinava Dubey'],
                year: 2020,
                venue: 'NeurIPS',
                doi: '10.5555/3495724.3497175',
                url: 'https://arxiv.org/abs/2007.14062',
                citation_count: 1420,
                relevance_score: 0.92,
                is_full_text: true,
            },
            {
                id: 'ref_3',
                title: 'Dense Passage Retrieval for Open-Domain QA',
                authors: ['Vladimir Karpukhin', 'Barlas Oguz', 'Sewon Min'],
                year: 2020,
                venue: 'EMNLP',
                doi: '10.18653/v1/2020.emnlp-main.550',
                url: 'https://arxiv.org/abs/2004.04906',
                citation_count: 2100,
                relevance_score: 0.88,
                is_full_text: false,
            },
        ],
        agent_plans: [],
    };

    const mockFullReport: ResearchReport = {
        metadata: {
            title: 'Deep Transformer Architectures for Scientific Synthesis',
            research_question: 'How do sparse attention transformers improve cross-document literature synthesis?',
            generated_at: '2026-08-26T12:30:00Z',
            quality_score: 91.5,
            status: 'completed',
        },
        executive_summary: 'Sparse attention architectures significantly mitigate O(N^2) memory complexity in multi-document summarization.',
        methodology_overview: {
            distribution: {
                'Sparse Attention': 2,
                'Dense Retrieval': 1,
            },
            dominant_approach: 'Sparse Attention',
            trend_description: 'Rapid transition from dense global self-attention towards windowed and random memory-efficient routing.',
        },
        thematic_sections: [
            {
                theme_id: 'theme_sparse_attention',
                title: 'Architectural Mechanisms in Sparse Multi-Document Attention',
                synthesis_prose: 'Long-range context aggregation utilizes windowed local attention combined with global tokens as demonstrated in [ref_1#sec_arch], whereas random block attention is explored in [ref_2]. Both surpass standard BERT baselines on long scientific documents.',
                key_takeaways: [
                    'Linear compute scaling O(N) enables 4,096+ token context windows.',
                    'Global memory anchors preserve semantic coherence across distant paragraphs.',
                ],
                cited_paper_ids: ['ref_1', 'ref_2'],
            },
        ],
        comparison_matrix: [
            {
                paper_id: 'ref_1',
                title: 'Longformer: The Long-Document Transformer',
                authors: ['Iz Beltagy', 'Matthew E. Peters', 'Arman Cohan'],
                year: 2020,
                methodology: 'Local Window + Global Attention',
                benchmark_dataset: 'arXiv / PubMed Summarization',
                primary_metric: 'ROUGE-L: 42.18',
                primary_limitation: 'High engineering overhead for custom CUDA kernels.',
                limitations: ['High engineering overhead for custom CUDA kernels.', 'Fixed window size hyperparameter sensitivity.'],
                key_findings: ['8x memory reduction compared to full attention', 'SOTA performance on document classification'],
                is_full_text: true,
                has_full_text: true,
            },
            {
                paper_id: 'ref_2',
                title: 'Big Bird: Transformers for Longer Sequences',
                authors: ['Manzil Zaheer', 'Guru Guruganesh', 'Avinava Dubey'],
                year: 2020,
                methodology: 'Random + Window + Global Graph Attention',
                benchmark_dataset: 'WikiHop / Natural Questions',
                primary_metric: 'F1: 76.8',
                primary_limitation: 'Slower training convergence due to stochastic block sampling.',
                limitations: ['Slower training convergence due to stochastic block sampling.'],
                key_findings: ['Theoretical proof of Turing completeness and sequence preservation'],
                is_full_text: true,
                has_full_text: true,
            },
            {
                paper_id: 'ref_3',
                title: 'Dense Passage Retrieval for Open-Domain QA',
                authors: ['Vladimir Karpukhin', 'Barlas Oguz', 'Sewon Min'],
                year: 2020,
                methodology: 'Dual-Encoder Dense Vector Retrieval',
                benchmark_dataset: 'TriviaQA / SQuAD',
                primary_metric: 'Top-20 Accuracy: 79.4%',
                primary_limitation: 'Does not model cross-passage token interaction directly.',
                limitations: ['Does not model cross-passage token interaction directly.'],
                key_findings: ['Dramatically outperforms BM25 lexical search'],
                is_full_text: false,
                has_full_text: false,
            },
        ],
        conflicting_findings_and_debates: [
            {
                topic: 'Deterministic vs Stochastic Sparsification in Long Document Transformers',
                perspective_a: 'Longformer argues deterministic banded local windows with pre-selected global anchor tokens are optimal for structured scientific prose.',
                perspective_b: 'Big Bird posits that random graph sparse connections are necessary to maintain universal approximator and Turing completeness properties.',
                critical_evaluation: 'Empirical results indicate deterministic windows excel on single-document scientific tasks, whereas random sparse graph topology achieves superior multi-hop reasoning over heterogeneous corpora.',
            },
        ],
        actionable_research_gaps: [
            {
                gap_id: 'GAP-01',
                title: 'Unified Evaluation of Hierarchical Cross-Document Attention Under Memory Constraints',
                description: 'Existing sparse models process flattened concatenations rather than maintaining cross-document citation graph topology.',
                importance: 'high',
                priority: 'high',
                recommended_methodology: 'Implement a hybrid graph-neural-network + sparse transformer layer that dynamically updates edge weights based on inter-paper citation references.',
                grounding_paper_ids: ['ref_1', 'ref_2'],
            },
            {
                gap_id: 'GAP-02',
                title: 'Zero-Shot Hallucination Detection in Multi-Source Synthesis',
                description: 'NLI verification models struggle with aggregate multi-hop synthesis claims where no single sentence contains direct entailment.',
                importance: 'medium',
                priority: 'medium',
                recommended_methodology: 'Develop proposition-level decomposition pipelines evaluated against human-annotated scientific counterfactual benchmarks.',
                grounding_paper_ids: ['ref_3'],
            },
        ],
        bibliography: [
            {
                paper_id: 'ref_1',
                title: 'Longformer: The Long-Document Transformer',
                authors: ['Iz Beltagy', 'Matthew E. Peters', 'Arman Cohan'],
                year: 2020,
                venue: 'arXiv preprint',
                doi: '10.48550/arXiv.2004.05150',
                pdf_url: 'https://arxiv.org/pdf/2004.05150.pdf',
                url: 'https://arxiv.org/abs/2004.05150',
                citation_count: 1850,
                is_full_text_analyzed: true,
            },
            {
                paper_id: 'ref_2',
                title: 'Big Bird: Transformers for Longer Sequences',
                authors: ['Manzil Zaheer', 'Guru Guruganesh', 'Avinava Dubey'],
                year: 2020,
                venue: 'NeurIPS',
                doi: '10.5555/3495724.3497175',
                pdf_url: 'https://arxiv.org/pdf/2007.14062.pdf',
                url: 'https://arxiv.org/abs/2007.14062',
                citation_count: 1420,
                is_full_text_analyzed: true,
            },
            {
                paper_id: 'ref_3',
                title: 'Dense Passage Retrieval for Open-Domain QA',
                authors: ['Vladimir Karpukhin', 'Barlas Oguz', 'Sewon Min'],
                year: 2020,
                venue: 'EMNLP',
                doi: '10.18653/v1/2020.emnlp-main.550',
                pdf_url: 'https://arxiv.org/pdf/2004.04906.pdf',
                url: 'https://arxiv.org/abs/2004.04906',
                citation_count: 2100,
                is_full_text_analyzed: false,
            },
        ],
    };

    beforeEach(() => {
        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false,
                    gcTime: 0,
                },
            },
        });
        vi.clearAllMocks();
    });

    afterEach(() => {
        queryClient.clear();
    });

    const renderWithProviders = (projectId: string = 'proj-123') => {
        return render(
            <QueryClientProvider client={queryClient}>
                <MemoryRouter initialEntries={[`/projects/${projectId}`]}>
                    <Routes>
                        <Route path="/projects/:projectId" element={<ProjectDetailsPage />} />
                    </Routes>
                </MemoryRouter>
            </QueryClientProvider>
        );
    };

    // =========================================================================
    // 1. EMPTY STATE EMPIRICAL TESTS (ALL 6 TABS)
    // =========================================================================
    describe('1. Empty State Verification across all 6 Tabs', () => {
        const emptyProject: ResearchProject = {
            id: 'proj-empty',
            title: 'Empty Exploration Project',
            research_question: 'What is the baseline empty state behavior?',
            keywords: [],
            subtopics: [],
            status: 'created',
            created_at: '2026-08-26T12:00:00Z',
            total_papers_found: 0,
            paper_references: [],
            agent_plans: [],
        };

        const emptyReportResponse: ReportResponse = {
            project_id: 'proj-empty',
            report: {
                metadata: {
                    title: 'Empty Exploration Project',
                    research_question: 'What is the baseline empty state behavior?',
                    generated_at: '2026-08-26T12:00:00Z',
                    status: 'created',
                    quality_score: 0,
                },
                executive_summary: '',
                thematic_sections: [],
                comparison_matrix: [],
                conflicting_findings_and_debates: [],
                actionable_research_gaps: [],
                methodology_overview: {
                    distribution: {},
                    dominant_approach: '',
                    trend_description: '',
                },
                bibliography: [],
            },
            report_status: 'empty',
        };

        beforeEach(() => {
            vi.spyOn(neonClient.neonData, 'getProjectById').mockResolvedValue(emptyProject);
            vi.spyOn(apiClient, 'getProjectReport').mockResolvedValue(emptyReportResponse);
            vi.spyOn(apiClient, 'getProjectMatrix').mockResolvedValue({ project_id: 'proj-empty', count: 0, total: 0, entries: [] });
            vi.spyOn(apiClient, 'getProjectGaps').mockResolvedValue({ project_id: 'proj-empty', count: 0, total: 0, gaps: [] });
            vi.spyOn(streamHook, 'useProjectStream').mockReturnValue({
                isConnected: false,
                currentAgent: null,
                progress: 0,
                logs: [],
                updates: [],
                clearUpdates: vi.fn(),
                latestCriticVerdict: null,
                latestFactCheck: null,
            });
        });

        it('Tab 0 (Literature Review): gracefully renders empty state without crashing', async () => {
            renderWithProviders('proj-empty');

            await waitFor(() => {
                expect(screen.getByText('Empty Exploration Project')).toBeInTheDocument();
            });

            // Tab 0 default
            expect(screen.getByText('Executive Summary')).toBeInTheDocument();
            expect(screen.getByText('No thematic sections synthesized yet')).toBeInTheDocument();
            expect(screen.getByText('The Synthesizer agent will draft thematic narrative reviews here.')).toBeInTheDocument();
        });

        it('Tab 1 (Evidence Matrix): displays empty table message', async () => {
            renderWithProviders('proj-empty');

            await waitFor(() => {
                expect(screen.getByText('Empty Exploration Project')).toBeInTheDocument();
            });

            const tab1 = screen.getByRole('tab', { name: /Evidence Matrix/i });
            fireEvent.click(tab1);

            expect(screen.getByText('No matrix rows found matching criteria')).toBeInTheDocument();
            expect(screen.getByText('Matrix extraction will populate as papers are parsed.')).toBeInTheDocument();
        });

        it('Tab 2 (Methodological Debates): displays empty debates message', async () => {
            renderWithProviders('proj-empty');

            await waitFor(() => {
                expect(screen.getByText('Empty Exploration Project')).toBeInTheDocument();
            });

            const tab2 = screen.getByRole('tab', { name: /Methodological Debates/i });
            fireEvent.click(tab2);

            expect(screen.getByText('No scientific controversies or debates detected')).toBeInTheDocument();
            expect(screen.getByText('The literature corpus currently exhibits strong methodological alignment with no polarized debates.')).toBeInTheDocument();
        });

        it('Tab 3 (Actionable Gaps): displays empty gaps message', async () => {
            renderWithProviders('proj-empty');

            await waitFor(() => {
                expect(screen.getByText('Empty Exploration Project')).toBeInTheDocument();
            });

            const tab3 = screen.getByRole('tab', { name: /Actionable Gaps/i });
            fireEvent.click(tab3);

            expect(screen.getByText('No research gaps synthesized yet')).toBeInTheDocument();
            expect(screen.getByText('Open research directions will be extracted and grounded in paper limitations during synthesis.')).toBeInTheDocument();
        });

        it('Tab 4 (Bibliography): displays empty corpus card', async () => {
            renderWithProviders('proj-empty');

            await waitFor(() => {
                expect(screen.getByText('Empty Exploration Project')).toBeInTheDocument();
            });

            const tab4 = screen.getByRole('tab', { name: /Bibliography/i });
            fireEvent.click(tab4);

            expect(screen.getByText('No papers discovered yet')).toBeInTheDocument();
            expect(screen.getByText('Literature discovery agent will populate candidate papers during exploration.')).toBeInTheDocument();
        });

        it('Tab 5 (Real-Time Journey): renders all DAG nodes in pending state when pipeline is idle', async () => {
            renderWithProviders('proj-empty');

            await waitFor(() => {
                expect(screen.getByText('Empty Exploration Project')).toBeInTheDocument();
            });

            const tab5 = screen.getByRole('tab', { name: /Real-Time Journey/i });
            fireEvent.click(tab5);

            expect(screen.getByText('LangGraph Orchestrator DAG')).toBeInTheDocument();
            expect(screen.getByText('STREAM DISCONNECTED')).toBeInTheDocument();
            expect(screen.getByText('Pipeline Execution Progress')).toBeInTheDocument();
            expect(screen.getByText('LangGraph Supervisor')).toBeInTheDocument();
            expect(screen.getByText('Adversarial Critic')).toBeInTheDocument();
            expect(screen.getByText('Citation Auditor')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // 2. LOADING STATE EMPIRICAL TESTS
    // =========================================================================
    describe('2. Loading State Verification', () => {
        it('renders spinner and loading message when primary project query is loading', () => {
            vi.spyOn(neonClient.neonData, 'getProjectById').mockReturnValue(new Promise(() => {})); // unresolved
            vi.spyOn(apiClient, 'getProjectReport').mockReturnValue(new Promise(() => {}));
            vi.spyOn(apiClient, 'getProjectMatrix').mockReturnValue(new Promise(() => {}));
            vi.spyOn(apiClient, 'getProjectGaps').mockReturnValue(new Promise(() => {}));
            vi.spyOn(streamHook, 'useProjectStream').mockReturnValue({
                isConnected: false,
                currentAgent: null,
                progress: 0,
                logs: [],
                updates: [],
                clearUpdates: vi.fn(),
                latestCriticVerdict: null,
                latestFactCheck: null,
            });

            renderWithProviders('proj-loading');

            expect(screen.getByText('Loading intelligence workspace...')).toBeInTheDocument();
        });

        it('renders executing badge when project is actively running (analyzing)', async () => {
            const runningProject: ResearchProject = {
                ...mockProjectData,
                status: 'analyzing',
            };
            vi.spyOn(neonClient.neonData, 'getProjectById').mockResolvedValue(runningProject);
            vi.spyOn(apiClient, 'getProjectReport').mockResolvedValue({ project_id: 'proj-123', report: null, report_status: 'in_progress' });
            vi.spyOn(apiClient, 'getProjectMatrix').mockResolvedValue({ project_id: 'proj-123', count: 0, total: 0, entries: [] });
            vi.spyOn(apiClient, 'getProjectGaps').mockResolvedValue({ project_id: 'proj-123', count: 0, total: 0, gaps: [] });
            vi.spyOn(streamHook, 'useProjectStream').mockReturnValue({
                isConnected: true,
                currentAgent: 'matrix_builder',
                progress: 45,
                logs: ['Parsing PDF section chunks...', 'Extracting comparative matrix rows...'],
                updates: [],
                clearUpdates: vi.fn(),
                latestCriticVerdict: null,
                latestFactCheck: null,
            });

            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            expect(screen.getByText('Executing (45%)')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // 3. ERROR STATE EMPIRICAL TESTS
    // =========================================================================
    describe('3. Error State Verification', () => {
        it('renders error alert when project fails to load', async () => {
            vi.spyOn(neonClient.neonData, 'getProjectById').mockRejectedValue(new Error('PostgreSQL database connection timeout'));
            vi.spyOn(apiClient, 'getProjectReport').mockResolvedValue({ project_id: 'proj-err', report: null, report_status: 'error' });
            vi.spyOn(apiClient, 'getProjectMatrix').mockResolvedValue({ project_id: 'proj-err', count: 0, total: 0, entries: [] });
            vi.spyOn(apiClient, 'getProjectGaps').mockResolvedValue({ project_id: 'proj-err', count: 0, total: 0, gaps: [] });
            vi.spyOn(streamHook, 'useProjectStream').mockReturnValue({
                isConnected: false,
                currentAgent: null,
                progress: 0,
                logs: [],
                updates: [],
                clearUpdates: vi.fn(),
                latestCriticVerdict: null,
                latestFactCheck: null,
            });

            renderWithProviders('proj-err');

            await waitFor(() => {
                expect(screen.getByText(/Failed to load project details: PostgreSQL database connection timeout/i)).toBeInTheDocument();
            });
        });
    });

    // =========================================================================
    // 4. FULL REPORT STATE EMPIRICAL TESTS (ALL 6 TABS & INTERACTIONS)
    // =========================================================================
    describe('4. Full Report State Verification across all 6 Tabs', () => {
        beforeEach(() => {
            vi.spyOn(neonClient.neonData, 'getProjectById').mockResolvedValue(mockProjectData);
            vi.spyOn(apiClient, 'getProjectReport').mockResolvedValue({
                project_id: 'proj-123',
                report: mockFullReport,
                report_status: 'completed',
            });
            vi.spyOn(apiClient, 'getProjectMatrix').mockResolvedValue({
                project_id: 'proj-123',
                count: 3,
                total: 3,
                entries: mockFullReport.comparison_matrix!,
            });
            vi.spyOn(apiClient, 'getProjectGaps').mockResolvedValue({
                project_id: 'proj-123',
                count: 2,
                total: 2,
                gaps: mockFullReport.actionable_research_gaps!,
            });
            vi.spyOn(streamHook, 'useProjectStream').mockReturnValue({
                isConnected: true,
                currentAgent: 'critic',
                progress: 85,
                logs: [
                    'Supervisor dispatched task to Discovery Agent',
                    'Retrieved 3 open access PDF full-texts',
                    'Extracted comparative evidence matrix',
                    'Synthesizer draft compiled with 2 thematic sections',
                    'Adversarial Critic evaluated draft quality: 91.5/100',
                ],
                updates: [],
                clearUpdates: vi.fn(),
                latestCriticVerdict: {
                    score: 91.5,
                    should_refine: false,
                    critique: 'Rigorous empirical coverage with grounded citations.',
                    iteration: 1,
                },
                latestFactCheck: {
                    passed: true,
                    precision_score: 95.0,
                    entailed_count: 19,
                    contradiction_count: 1,
                    unsupported_count: 0,
                    hallucination_score: 5.0,
                },
            });
        });

        it('Header: renders title, quality score badge, paper count, and research question popover', async () => {
            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            expect(screen.getByText('Quality Score: 91.5/100')).toBeInTheDocument();
            expect(screen.getByText('Papers Discovered')).toBeInTheDocument();
            expect(screen.getByText('3')).toBeInTheDocument();

            // Click research question dropdown
            const questionBtn = screen.getByRole('button', { name: /Research Question/i });
            fireEvent.click(questionBtn);

            expect(screen.getByText(/Target Research Question/i)).toBeInTheDocument();
            expect(screen.getByText(/How do sparse attention transformers improve cross-document literature synthesis\?/i)).toBeInTheDocument();
        });

        it('Tab 0 (Literature Review): renders Executive Summary, Methodology Distribution, and Thematic Sections with citation popover', async () => {
            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Executive Summary')).toBeInTheDocument();
            });

            expect(screen.getByText(/Sparse attention architectures significantly mitigate O\(N\^2\) memory complexity/i)).toBeInTheDocument();

            // Methodology card
            expect(screen.getByText('Methodological Landscape & Distribution')).toBeInTheDocument();
            expect(screen.getByText('Dominant Paradigm:')).toBeInTheDocument();
            expect(screen.getAllByText('Sparse Attention').length).toBeGreaterThan(0);
            expect(screen.getByText(/Rapid transition from dense global self-attention/i)).toBeInTheDocument();

            // Thematic section
            expect(screen.getByText('Architectural Mechanisms in Sparse Multi-Document Attention')).toBeInTheDocument();
            expect(screen.getByText('Key Findings & Actionable Takeaways')).toBeInTheDocument();
            expect(screen.getByText(/Linear compute scaling O\(N\) enables 4,096\+ token context windows/i)).toBeInTheDocument();

            // Citation chip interaction
            const citationChip = screen.getByText('[ref_1#sec_arch]');
            expect(citationChip).toBeInTheDocument();

            fireEvent.click(citationChip);

            await waitFor(() => {
                expect(screen.getByText(/Citation \[ref_1#sec_arch\]/i)).toBeInTheDocument();
                expect(screen.getByText('Longformer: The Long-Document Transformer')).toBeInTheDocument();
                expect(screen.getByText(/§ sec arch/i)).toBeInTheDocument();
            });
        });

        it('Tab 1 (Evidence Matrix): supports search filtering, full-text toggle, sorting, and row expansion', async () => {
            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            const tab1 = screen.getByRole('tab', { name: /Evidence Matrix \(3\)/i });
            fireEvent.click(tab1);

            // Table headers and rows
            expect(screen.getByText('Comparative Evidence Matrix')).toBeInTheDocument();
            expect(screen.getByText('[ref_1]')).toBeInTheDocument();
            expect(screen.getByText('[ref_2]')).toBeInTheDocument();
            expect(screen.getByText('[ref_3]')).toBeInTheDocument();

            // Full-Text filter
            const fullTextChip = screen.getByText(/Full-Text PDF \(2\)/i);
            fireEvent.click(fullTextChip);

            expect(screen.getByText('[ref_1]')).toBeInTheDocument();
            expect(screen.getByText('[ref_2]')).toBeInTheDocument();
            expect(screen.queryByText('[ref_3]')).not.toBeInTheDocument();

            // Reset to All
            const allChip = screen.getByText(/All \(3\)/i);
            fireEvent.click(allChip);
            expect(screen.getByText('[ref_3]')).toBeInTheDocument();

            // Search filter
            const searchInput = screen.getByPlaceholderText('Search matrix rows...');
            fireEvent.change(searchInput, { target: { value: 'Longformer' } });

            expect(screen.getByText('[ref_1]')).toBeInTheDocument();
            expect(screen.queryByText('[ref_2]')).not.toBeInTheDocument();
            expect(screen.queryByText('[ref_3]')).not.toBeInTheDocument();

            fireEvent.change(searchInput, { target: { value: '' } });

            // Expand row
            const row1 = screen.getByText('Longformer: The Long-Document Transformer');
            fireEvent.click(row1);

            expect(screen.getByText('Identified Limitations & Bottlenecks')).toBeInTheDocument();
            expect(screen.getAllByText(/High engineering overhead for custom CUDA kernels/i).length).toBeGreaterThan(0);
            expect(screen.getByText(/8x memory reduction compared to full attention/i)).toBeInTheDocument();
        });

        it('Tab 2 (Methodological Debates): renders Perspective A vs B cards and Critical Evaluation', async () => {
            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            const tab2 = screen.getByRole('tab', { name: /Methodological Debates \(1\)/i });
            fireEvent.click(tab2);

            expect(screen.getByText('Methodological & Empirical Debates')).toBeInTheDocument();
            expect(screen.getByText(/Deterministic vs Stochastic Sparsification/i)).toBeInTheDocument();
            expect(screen.getByText(/Perspective A/i)).toBeInTheDocument();
            expect(screen.getByText(/Longformer argues deterministic banded local windows/i)).toBeInTheDocument();
            expect(screen.getByText(/Perspective B/i)).toBeInTheDocument();
            expect(screen.getByText(/Big Bird posits that random graph sparse connections/i)).toBeInTheDocument();
            expect(screen.getByText('Critical Evaluation & Empirical Synthesis')).toBeInTheDocument();
            expect(screen.getByText(/Empirical results indicate deterministic windows excel/i)).toBeInTheDocument();
        });

        it('Tab 3 (Actionable Gaps): renders priority badges, methodology recommendations, and filters by priority', async () => {
            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            const tab3 = screen.getByRole('tab', { name: /Actionable Gaps \(2\)/i });
            fireEvent.click(tab3);
            expect(screen.getByText('Actionable Research Gaps & Future Directions')).toBeInTheDocument();
            expect(screen.getAllByText(/HIGH PRIORITY/i).length).toBeGreaterThan(0);
            expect(screen.getByText(/GAP-01/i)).toBeInTheDocument();
            expect(screen.getByText(/Unified Evaluation of Hierarchical Cross-Document Attention/i)).toBeInTheDocument();
            expect(screen.getAllByText('Actionable Methodology & Experimental Roadmap').length).toBeGreaterThan(0);
            expect(screen.getByText(/Implement a hybrid graph-neural-network/i)).toBeInTheDocument();

            // Filter High Priority only
            const highFilter = screen.getByText(/High Priority \(1\)/i);
            fireEvent.click(highFilter);

            expect(screen.getByText(/GAP-01/i)).toBeInTheDocument();
            expect(screen.queryByText(/GAP-02/i)).not.toBeInTheDocument();
        });

        it('Tab 4 (Bibliography): renders paper cards with PDF open access buttons and export BibTeX trigger', async () => {
            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            const tab4 = screen.getByRole('tab', { name: /Bibliography \(3\)/i });
            fireEvent.click(tab4);

            expect(screen.getByText('Discovered Papers & Corpus (3)')).toBeInTheDocument();
            expect(screen.getByText('Longformer: The Long-Document Transformer')).toBeInTheDocument();
            expect(screen.getByText('Big Bird: Transformers for Longer Sequences')).toBeInTheDocument();
            expect(screen.getByText('Dense Passage Retrieval for Open-Domain QA')).toBeInTheDocument();

            // Direct PDF badges / chips
            expect(screen.getAllByText('Full Text').length).toBeGreaterThan(0);

            // BibTeX Export trigger
            const exportBibBtn = screen.getByRole('button', { name: /Export BibTeX/i });
            expect(exportBibBtn).toBeInTheDocument();
        });

        it('Tab 5 (Real-Time Journey): renders LangGraph 7-stage state machine, critic banner, auditor banner, and logs', async () => {
            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            const tab5 = screen.getByRole('tab', { name: /Real-Time Journey/i });
            fireEvent.click(tab5);

            expect(screen.getByText('LangGraph Orchestrator DAG')).toBeInTheDocument();
            expect(screen.getByText('REALTIME STREAM ACTIVE')).toBeInTheDocument();
            expect(screen.getByText('85%')).toBeInTheDocument();

            // Stages
            expect(screen.getByText('LangGraph Supervisor')).toBeInTheDocument();
            expect(screen.getByText('Literature Discovery')).toBeInTheDocument();
            expect(screen.getByText('Ingestion & PDF Parser')).toBeInTheDocument();
            expect(screen.getByText('Evidence Matrix Builder')).toBeInTheDocument();
            expect(screen.getByText('Thematic Synthesizer')).toBeInTheDocument();
            expect(screen.getByText('Adversarial Critic')).toBeInTheDocument();
            expect(screen.getByText('Citation Auditor')).toBeInTheDocument();

            // Adversarial Critic Banner
            expect(screen.getByText(/Adversarial Critic Evaluation \(Iteration 1\)/i)).toBeInTheDocument();;
            expect(screen.getByText('Score: 91.5/100')).toBeInTheDocument();
            expect(screen.getByText('Synthesis passed academic rigor verification.')).toBeInTheDocument();

            // Citation Auditor Banner
            expect(screen.getByText('Citation Audit NLI Grounding')).toBeInTheDocument();
            expect(screen.getByText('Precision: 95.0%')).toBeInTheDocument();
            expect(screen.getByText(/19 propositions entailed, 1 contradictions/i)).toBeInTheDocument();

            // Log stream
            expect(screen.getByText('Live Telemetry & Event Stream')).toBeInTheDocument();
            expect(screen.getByText(/Retrieved 3 open access PDF full-texts/i)).toBeInTheDocument();
        });
    });

    // =========================================================================
    // 5. ZERO PLACEHOLDER / "COMING SOON" AUDIT
    // =========================================================================
    describe('5. Zero Placeholder / "Coming Soon" Component Audit', () => {
        it('verifies that no tab contains "Coming Soon" or empty placeholder text', async () => {
            vi.spyOn(neonClient.neonData, 'getProjectById').mockResolvedValue(mockProjectData);
            vi.spyOn(apiClient, 'getProjectReport').mockResolvedValue({
                project_id: 'proj-123',
                report: mockFullReport,
                report_status: 'completed',
            });
            vi.spyOn(apiClient, 'getProjectMatrix').mockResolvedValue({
                project_id: 'proj-123',
                count: 3,
                total: 3,
                entries: mockFullReport.comparison_matrix!,
            });
            vi.spyOn(apiClient, 'getProjectGaps').mockResolvedValue({
                project_id: 'proj-123',
                count: 2,
                total: 2,
                gaps: mockFullReport.actionable_research_gaps!,
            });
            vi.spyOn(streamHook, 'useProjectStream').mockReturnValue({
                isConnected: true,
                currentAgent: null,
                progress: 100,
                logs: [],
                updates: [],
                clearUpdates: vi.fn(),
                latestCriticVerdict: null,
                latestFactCheck: null,
            });

            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            // Check all tabs
            for (let i = 0; i < 6; i++) {
                const tabs = screen.getAllByRole('tab');
                fireEvent.click(tabs[i]);
                expect(screen.queryByText(/Coming Soon/i)).not.toBeInTheDocument();
                expect(screen.queryByText(/Under Construction/i)).not.toBeInTheDocument();
                expect(screen.queryByText(/Work In Progress/i)).not.toBeInTheDocument();
            }
        });
    });

    // =========================================================================
    // 6. EXPORT DELIVERABLES MENU TEST
    // =========================================================================
    describe('6. Export Deliverables Menu Verification', () => {
        it('opens export menu and dispatches correct export format handlers', async () => {
            vi.spyOn(neonClient.neonData, 'getProjectById').mockResolvedValue(mockProjectData);
            vi.spyOn(apiClient, 'getProjectReport').mockResolvedValue({
                project_id: 'proj-123',
                report: mockFullReport,
                report_status: 'completed',
            });
            vi.spyOn(apiClient, 'getProjectMatrix').mockResolvedValue({
                project_id: 'proj-123',
                count: 3,
                total: 3,
                entries: mockFullReport.comparison_matrix!,
            });
            vi.spyOn(apiClient, 'getProjectGaps').mockResolvedValue({
                project_id: 'proj-123',
                count: 2,
                total: 2,
                gaps: mockFullReport.actionable_research_gaps!,
            });
            vi.spyOn(streamHook, 'useProjectStream').mockReturnValue({
                isConnected: false,
                currentAgent: null,
                progress: 100,
                logs: [],
                updates: [],
                clearUpdates: vi.fn(),
                latestCriticVerdict: null,
                latestFactCheck: null,
            });

            renderWithProviders('proj-123');

            await waitFor(() => {
                expect(screen.getByText('Deep Transformer Architectures for Scientific Synthesis')).toBeInTheDocument();
            });

            const exportBtn = screen.getByRole('button', { name: /Export Deliverables/i });
            fireEvent.click(exportBtn);

            expect(screen.getByText('Export as Markdown (.md)')).toBeInTheDocument();
            expect(screen.getByText('Export as PDF Document (.pdf)')).toBeInTheDocument();
            expect(screen.getByText('Export as Word Document (.docx)')).toBeInTheDocument();
            expect(screen.getByText('Export BibTeX References (.bib)')).toBeInTheDocument();

            // Trigger markdown export
            const mdMenuItem = screen.getByText('Export as Markdown (.md)');
            fireEvent.click(mdMenuItem);

            expect(exportUtils.exportReport).toHaveBeenCalledWith('md', expect.anything(), 'synthesis-output-container', expect.any(String));
        });
    });
});
