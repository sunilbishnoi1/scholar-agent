/**
 * Comprehensive Empirical Stress Test Suite for Milestone 4 (Challenger 2)
 *
 * Covers:
 * 1. Export Engine Edge Cases (empty, special chars, pipes, nulls, bibtex, pdf DOM, docx)
 * 2. Rapid WebSocket Event Stream Ingestion & State Invariants
 * 3. Component Rendering Resilience Under Edge-Case & Adversarial Inputs
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Export Engine functions
import {
    formatAuthors,
    slugifyTitle,
    generateExportFilename,
    generateReportMarkdown,
    generateBibTeX,
    createAcademicPrintableContainer,
    exportToPdfDirect,
    exportToDocx,
} from '../utils/exportEngine';

// Dashboard Components
import { ConflictingDebates } from '../components/dashboard/ConflictingDebates';
import { EvidenceMatrixTable } from '../components/dashboard/EvidenceMatrixTable';
import { ResearchGapViewer } from '../components/dashboard/ResearchGapViewer';
import { MethodologyDistributionCard } from '../components/dashboard/MethodologyDistributionCard';
import { ThematicSections } from '../components/dashboard/ThematicSections';

// Hook & Types
import { useProjectStream } from '../hooks/useProjectStream';
import type {
    ResearchReport,
    EvidenceMatrixRow,
    ConflictingDebate,
    ResearchGapItem,
    MethodologyDistribution,
    ThematicSection,
    BibliographyItem,
} from '../types';

describe('M4 Challenger 2 Empirical Stress Test Suite', () => {

    // =========================================================================
    // PART 1: EXPORT ENGINE EDGE CASES & ADVERSARIAL INPUTS
    // =========================================================================
    describe('1. Export Engine Edge Cases & Formatting Resilience', () => {

        describe('formatAuthors edge cases', () => {
            it('handles null, undefined, empty array, and non-array types', () => {
                expect(formatAuthors(undefined)).toBe('Unknown Authors');
                expect(formatAuthors(null)).toBe('Unknown Authors');
                expect(formatAuthors([])).toBe('Unknown Authors');
                // @ts-expect-error test invalid type
                expect(formatAuthors('Not an array')).toBe('Unknown Authors');
                // @ts-expect-error test invalid type
                expect(formatAuthors(123)).toBe('Unknown Authors');
            });

            it('filters out placeholders like "Unknown", "unknown authors", and empty strings', () => {
                expect(formatAuthors(['Unknown'])).toBe('Unknown Authors');
                expect(formatAuthors(['unknown authors', '   '])).toBe('Unknown Authors');
                expect(formatAuthors(['', 'Alice Smith', 'Unknown'])).toBe('Alice Smith');
            });

            it('handles unicode, international characters, and accents', () => {
                expect(formatAuthors(['José López-García', 'François Müller'])).toBe('José López-García and François Müller');
                expect(formatAuthors(['山田 太郎', 'Иван Иванов', 'Müller-Thurgau'])).toBe('山田 太郎 et al.');
            });

            it('handles 1, 2, and 3+ authors correctly', () => {
                expect(formatAuthors(['Jane Doe'])).toBe('Jane Doe');
                expect(formatAuthors(['Jane Doe', 'John Smith'])).toBe('Jane Doe and John Smith');
                expect(formatAuthors(['Jane Doe', 'John Smith', 'Alan Turing', 'Ada Lovelace'])).toBe('Jane Doe et al.');
            });
        });

        describe('slugifyTitle & filename generation', () => {
            it('handles empty, whitespace-only, and punctuation-only titles', () => {
                expect(slugifyTitle('')).toBe('literature_review');
                expect(slugifyTitle('    ')).toBe('literature_review');
                expect(slugifyTitle('!@#$%^&*()_+=-')).toBe('literature_review');
            });

            it('handles long strings, trimming to 45 chars max and stripping leading/trailing underscores', () => {
                const longTitle = 'An Ultra Comprehensive and Extremely Detailed Empirical Survey on Artificial Intelligence & LLMs in 2026';
                const slug = slugifyTitle(longTitle);
                expect(slug.length).toBeLessThanOrEqual(45);
                expect(slug).not.toMatch(/^_+|_+$/);
                expect(slug).toBe('an_ultra_comprehensive_and_extremely_detailed');
            });

            it('generates filename with correct extension and date format', () => {
                const fn = generateExportFilename('Special & Punctuation! Test', 'docx');
                expect(fn).toMatch(/^special_punctuation_test_\d{4}-\d{2}-\d{2}\.docx$/);
            });
        });

        describe('generateReportMarkdown edge cases', () => {
            it('handles completely bare report object with missing optional fields', () => {
                const bareReport: ResearchReport = {
                    title: 'Bare Minimum Report',
                };
                const md = generateReportMarkdown(bareReport);
                expect(md).toContain('# Bare Minimum Report');
                expect(md).toContain('**Research Question:** N/A');
                expect(md).toContain('**Quality Score:** N/A');
                expect(md).toContain('No executive summary generated.');
                expect(md).not.toContain('## Evidence Comparison Matrix');
                expect(md).not.toContain('## Thematic Synthesis');
                expect(md).not.toContain('## Conflicting Findings');
                expect(md).not.toContain('## Actionable Research Gaps');
                expect(md).not.toContain('## Bibliography');
            });

            it('escapes pipe "|" characters and newlines in Evidence Matrix table cells', () => {
                const reportWithPipes: ResearchReport = {
                    metadata: { title: 'Pipe Test' },
                    comparison_matrix: [
                        {
                            paper_id: 'ref_pipe',
                            title: 'Paper with | Pipe | Symbols',
                            authors: ['Author A', 'Author | B'],
                            year: 2025,
                            methodology: 'Method | A / B\nMulti-line',
                            benchmark_dataset: 'Dataset | Bench',
                            primary_metric: 'Metric | F1: 90%',
                            primary_limitation: 'Limitation | with | pipes\nand newlines',
                            is_full_text: true,
                        },
                    ],
                };
                const md = generateReportMarkdown(reportWithPipes);
                expect(md).toContain('## Evidence Comparison Matrix');
                // Ensure pipes inside table cells are escaped as \| and newlines replaced with space
                expect(md).toContain('Paper with \\| Pipe \\| Symbols');
                expect(md).toContain('Method \\| A / B Multi-line');
                expect(md).toContain('Limitation \\| with \\| pipes and newlines');
            });

            it('renders all sections when full data is present', () => {
                const fullReport: ResearchReport = {
                    metadata: {
                        title: 'Full Scientific Review',
                        research_question: 'What is the efficacy of agentic workflows?',
                        generated_at: '2026-08-28T00:00:00Z',
                        quality_score: 95.5,
                        status: 'COMPLETED',
                    },
                    executive_summary: 'Comprehensive analysis of 15 agents.',
                    methodology_overview: {
                        dominant_approach: 'Graph-based Multi-Agent',
                        trend_description: 'Exponential shift toward LangGraph supervisor topologies.',
                        distribution: { 'Graph-based': 10, 'Linear': 5 },
                    },
                    thematic_sections: [
                        {
                            theme_id: 't1',
                            title: 'Supervisor Coordination',
                            synthesis_prose: 'Supervisors coordinate specialist subagents effectively.',
                            key_takeaways: ['Takeaway 1', 'Takeaway 2'],
                        },
                    ],
                    conflicting_findings_and_debates: [
                        {
                            topic: 'Centralized vs Decentralized Routing',
                            perspective_a: 'Centralized supervisor is more predictable.',
                            perspective_b: 'Decentralized blackboard is more scalable.',
                            critical_evaluation: 'Centralized supervisor with blackboard memory achieves best accuracy.',
                        },
                    ],
                    actionable_research_gaps: [
                        {
                            gap_id: 'GAP-01',
                            title: 'Dynamic Agent Spawning',
                            description: 'Lack of self-adapting agent topologies under load.',
                            importance: 'HIGH',
                            recommended_methodology: 'Implement dynamic worker pool heuristics.',
                            grounding_paper_ids: ['ref_1', 'ref_2'],
                        },
                    ],
                    bibliography: [
                        {
                            paper_id: 'ref_1',
                            title: 'Multi-Agent Frameworks',
                            authors: ['Alice Doe', 'Bob Ray'],
                            year: 2024,
                            venue: 'ICLR',
                            doi: '10.1234/iclr.2024.001',
                            pdf_url: 'https://example.com/paper.pdf',
                            is_full_text_analyzed: true,
                        },
                    ],
                };

                const md = generateReportMarkdown(fullReport);
                expect(md).toContain('# Full Scientific Review');
                expect(md).toContain('**Quality Score:** 95.5/100');
                expect(md).toContain('## Methodology Overview');
                expect(md).toContain('**Dominant Approach:** Graph-based Multi-Agent');
                expect(md).toContain('- **Graph-based**: 10 papers');
                expect(md).toContain('## Thematic Synthesis');
                expect(md).toContain('### Supervisor Coordination');
                expect(md).toContain('- Takeaway 1');
                expect(md).toContain('## Conflicting Findings & Scientific Debates');
                expect(md).toContain('### Debate: Centralized vs Decentralized Routing');
                expect(md).toContain('## Actionable Research Gaps & Future Directions');
                expect(md).toContain('### [HIGH PRIORITY] GAP-01: Dynamic Agent Spawning');
                expect(md).toContain('## Bibliography');
                expect(md).toContain('**[ref_1]** Alice Doe and Bob Ray (2024). **Multi-Agent Frameworks**');
                expect(md).toContain('DOI: [10.1234/iclr.2024.001](https://doi.org/10.1234/iclr.2024.001)');
            });
        });

        describe('generateBibTeX edge cases', () => {
            it('handles empty items array', () => {
                expect(generateBibTeX([])).toBe('');
            });

            it('sanitizes keys with special characters and formats bibtex fields correctly', () => {
                const items: BibliographyItem[] = [
                    {
                        paper_id: 'arxiv:2024.12345v1/special#tag',
                        title: 'A Study on Robustness & Generalization',
                        authors: ['Alice Smith', 'Bob Jones', 'Charlie Brown'],
                        year: 2024,
                        venue: 'NeurIPS 2024',
                        doi: '10.1000/182',
                        pdf_url: 'https://arxiv.org/pdf/2024.12345.pdf',
                    },
                    {
                        paper_id: 'ref_2',
                        title: 'Paper Without Optional Metadata',
                        authors: [],
                    },
                ];

                const bib = generateBibTeX(items);
                expect(bib).toContain('@article{arxiv_2024_12345v1_special_tag,');
                expect(bib).toContain('title={A Study on Robustness & Generalization}');
                expect(bib).toContain('author={Alice Smith and Bob Jones and Charlie Brown}');
                expect(bib).toContain('journal={NeurIPS 2024}');
                expect(bib).toContain('doi={10.1000/182}');
                expect(bib).toContain('url={https://arxiv.org/pdf/2024.12345.pdf}');

                expect(bib).toContain('@article{ref_2,');
                expect(bib).toContain('title={Paper Without Optional Metadata}');
                expect(bib).toContain('author={Unknown}');
            });
        });

        describe('createAcademicPrintableContainer HTML generation', () => {
            it('creates DOM container for empty report without throwing', () => {
                const container = createAcademicPrintableContainer({});
                expect(container).toBeInstanceOf(HTMLDivElement);
                expect(container.id).toBe('academic-pdf-export-root');
                expect(container.innerHTML).toContain('Autonomous Scientific Literature Review');
                expect(container.innerHTML).toContain('Critic Quality Score: 85.0/100');
            });

            it('renders all sections and applies correct styling for populated report', () => {
                const report: ResearchReport = {
                    metadata: {
                        title: 'Printable Test Report',
                        research_question: 'How to evaluate PDF exports?',
                        quality_score: 88.0,
                    },
                    executive_summary: 'Testing printable DOM structure.',
                    comparison_matrix: [
                        {
                            paper_id: 'p1',
                            title: 'Matrix Title',
                            authors: ['Author One'],
                            year: 2023,
                            methodology: 'Test Method',
                            benchmark_dataset: 'Bench A',
                            primary_metric: 'Acc: 90%',
                            primary_limitation: 'High memory',
                        },
                    ],
                    thematic_sections: [
                        {
                            theme_id: 'sec_1',
                            title: 'Theme A',
                            synthesis_prose: 'Narrative content.',
                            key_takeaways: ['Takeaway A'],
                        },
                    ],
                    conflicting_findings_and_debates: [
                        {
                            topic: 'Debate Topic A',
                            perspective_a: 'Perspective 1',
                            perspective_b: 'Perspective 2',
                            critical_evaluation: 'Evaluation text.',
                        },
                    ],
                    actionable_research_gaps: [
                        {
                            gap_id: 'GAP-1',
                            description: 'Gap description',
                            importance: 'high',
                            recommended_methodology: 'Methodology roadmap',
                        },
                    ],
                    bibliography: [
                        {
                            paper_id: 'p1',
                            title: 'Matrix Title',
                            authors: ['Author One'],
                            year: 2023,
                            doi: '10.1234/test',
                        },
                    ],
                };

                const container = createAcademicPrintableContainer(report);
                expect(container.innerHTML).toContain('Printable Test Report');
                expect(container.innerHTML).toContain('Evidence Comparison Matrix');
                expect(container.innerHTML).toContain('Theme A');
                expect(container.innerHTML).toContain('Debate: Debate Topic A');
                expect(container.innerHTML).toContain('HIGH PRIORITY');
                expect(container.innerHTML).toContain('References &amp; Bibliography');
            });
        });

        describe('exportToPdfDirect & exportToDocx headless execution', () => {
            it('exportToPdfDirect executes jsPDF generation without crashing on minimal report', () => {
                const report: ResearchReport = {
                    metadata: { title: 'Direct PDF Test' },
                    executive_summary: 'Summary paragraph.',
                };
                expect(() => exportToPdfDirect(report, 'test.pdf')).not.toThrow();
            });

            it('exportToDocx builds docx blob without crashing', async () => {
                const report: ResearchReport = {
                    metadata: { title: 'Docx Test' },
                    comparison_matrix: [
                        {
                            paper_id: 'doc_1',
                            title: 'Docx Paper',
                            authors: ['Author X'],
                            year: 2024,
                            methodology: 'Method X',
                            primary_metric: 'Metric X',
                            primary_limitation: 'Limitation X',
                        },
                    ],
                    thematic_sections: [
                        {
                            theme_id: 't_doc',
                            title: 'Thematic Docx Section',
                            synthesis_prose: 'Synthesis prose text.',
                            key_takeaways: ['Takeaway docx'],
                        },
                    ],
                };

                await expect(exportToDocx(report, 'test.docx')).resolves.not.toThrow();
            });
        });
    });

    // =========================================================================
    // PART 2: REAL-TIME WEBSOCKET STREAM INGESTION & STATE INTEGRITY
    // =========================================================================
    describe('2. Real-Time WebSocket State Integrity & Rapid Message Ingestion', () => {
        let queryClient: QueryClient;

        // Custom Mock WebSocket for testing message ingestion
        class MockWebSocket {
            static instances: MockWebSocket[] = [];
            url: string;
            readyState: number = WebSocket.OPEN;
            onopen: ((event: unknown) => void) | null = null;
            onmessage: ((event: { data: string }) => void) | null = null;
            onclose: ((event: unknown) => void) | null = null;
            onerror: ((event: unknown) => void) | null = null;

            constructor(url: string) {
                this.url = url;
                MockWebSocket.instances.push(this);
                setTimeout(() => {
                    if (this.onopen) this.onopen({});
                }, 10);
            }

            send = vi.fn();
            close = vi.fn(() => {
                this.readyState = WebSocket.CLOSED;
                if (this.onclose) this.onclose({});
            });

            simulateMessage(data: unknown) {
                if (this.onmessage) {
                    this.onmessage({ data: JSON.stringify(data) });
                }
            }

            simulateRawMessage(rawData: string) {
                if (this.onmessage) {
                    this.onmessage({ data: rawData });
                }
            }
        }

        beforeEach(() => {
            queryClient = new QueryClient({
                defaultOptions: { queries: { retry: false, gcTime: 0 } },
            });
            MockWebSocket.instances = [];
            vi.stubGlobal('WebSocket', MockWebSocket);
        });

        afterEach(() => {
            vi.unstubAllGlobals();
            queryClient.clear();
        });

        const wrapper = ({ children }: { children: React.ReactNode }) => (
            <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
        );

        it('connects to WebSocket with project ID and token', async () => {
            const { result } = renderHook(
                () => useProjectStream('proj-realtime-1', { token: 'mock-token-123' }),
                { wrapper }
            );

            await waitFor(() => {
                expect(result.current.isConnected).toBe(true);
            });

            expect(MockWebSocket.instances.length).toBe(1);
            expect(MockWebSocket.instances[0].url).toContain('/ws/projects/proj-realtime-1/stream');
            expect(MockWebSocket.instances[0].url).toContain('token=mock-token-123');
        });

        it('ingests rapid burst of 60+ multi-agent DAG events while maintaining buffer invariants', async () => {
            const { result } = renderHook(
                () => useProjectStream('proj-burst', { autoReconnect: false }),
                { wrapper }
            );

            await waitFor(() => {
                expect(result.current.isConnected).toBe(true);
            });

            const ws = MockWebSocket.instances[0];

            // Rapid message ingestion burst
            act(() => {
                // 1. Discovery started
                ws.simulateMessage({
                    type: 'discovery_started',
                    agent: 'discovery',
                    progress: 10,
                    message: 'Initiating arXiv & Semantic Scholar discovery queries...',
                });

                // 2. 20 papers discovered in rapid succession
                for (let i = 1; i <= 20; i++) {
                    ws.simulateMessage({
                        type: 'paper_discovered',
                        agent: 'discovery',
                        progress: 10 + i,
                        message: `Discovered candidate paper #${i}: Attention Mechanism Analysis ${i}`,
                        data: { paper_id: `ref_${i}`, title: `Paper ${i}` },
                    });
                }

                // 3. 15 PDFs parsed
                for (let i = 1; i <= 15; i++) {
                    ws.simulateMessage({
                        type: 'pdf_parsed',
                        agent: 'ingestion',
                        progress: 30 + i,
                        message: `Parsed open-access PDF for paper #${i}`,
                        data: { paper_id: `ref_${i}`, chunks: 25 },
                    });
                }

                // 4. Matrix rows added
                for (let i = 1; i <= 10; i++) {
                    ws.simulateMessage({
                        type: 'matrix_row_added',
                        agent: 'matrix_builder',
                        progress: 45 + i,
                        message: `Extracted comparative evidence matrix row #${i}`,
                    });
                }

                // 5. Thematic draft ready
                ws.simulateMessage({
                    type: 'thematic_draft_ready',
                    agent: 'synthesizer',
                    progress: 65,
                    message: 'Compiled 3 thematic narrative synthesis sections',
                });

                // 6. Critic verdict
                ws.simulateMessage({
                    type: 'critic_verdict',
                    agent: 'critic',
                    progress: 80,
                    message: 'Adversarial Critic evaluated synthesis quality: 92.5/100',
                    data: {
                        score: 92.5,
                        should_refine: false,
                        iteration: 1,
                        dimension_scores: { clarity: 95, grounding: 90 },
                        weaknesses: ['Minor notation variance in section 2'],
                        guidance: 'Approve draft for final citation audit.',
                    },
                });

                // 7. Citation Auditor Fact Check
                ws.simulateMessage({
                    type: 'fact_checked',
                    agent: 'auditor',
                    progress: 95,
                    message: 'Citation Auditor completed NLI entailment verification',
                    data: {
                        precision_score: 96.0,
                        passed: true,
                        entailed_count: 24,
                        neutral_count: 1,
                        contradiction_count: 0,
                        total_propositions: 25,
                    },
                });

                // 8. Pipeline complete
                ws.simulateMessage({
                    type: 'pipeline_completed',
                    progress: 100,
                    message: 'Autonomous research pipeline successfully completed all stages.',
                });
            });

            // Verify state invariants
            expect(result.current.progress).toBe(100);
            expect(result.current.currentAgent).toBe(null); // Cleared on completion
            expect(result.current.totalPapers).toBe(20);
            expect(result.current.papersAnalyzed).toBe(15);

            // Verify critic details
            expect(result.current.latestCriticVerdict).toEqual({
                score: 92.5,
                should_refine: false,
                iteration: 1,
                dimension_scores: { clarity: 95, grounding: 90 },
                weaknesses: ['Minor notation variance in section 2'],
                guidance: 'Approve draft for final citation audit.',
            });

            // Verify auditor fact check details
            expect(result.current.latestFactCheck).toEqual({
                precision_score: 96.0,
                passed: true,
                entailed_count: 24,
                neutral_count: 1,
                contradiction_count: 0,
                total_propositions: 25,
            });

            // Verify log buffer caps (last 50 messages)
            expect(result.current.logs.length).toBeLessThanOrEqual(50);
            expect(result.current.logs[result.current.logs.length - 1]).toBe(
                'Autonomous research pipeline successfully completed all stages.'
            );

            // Verify update buffer caps (last 100 updates)
            expect(result.current.updates.length).toBeLessThanOrEqual(100);
        });

        it('ignores pong messages and gracefully handles malformed JSON without crashing', async () => {
            const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
            const { result } = renderHook(
                () => useProjectStream('proj-robustness', { autoReconnect: false }),
                { wrapper }
            );

            await waitFor(() => {
                expect(result.current.isConnected).toBe(true);
            });

            const ws = MockWebSocket.instances[0];

            act(() => {
                // Pong message should be skipped
                ws.simulateMessage({ type: 'pong' });
                // Malformed raw string
                ws.simulateRawMessage('{ broken json ...');
                // Valid message
                ws.simulateMessage({
                    type: 'log',
                    message: 'Valid log message after malformed input',
                    progress: 25,
                });
            });

            expect(result.current.progress).toBe(25);
            expect(result.current.logs).toContain('Valid log message after malformed input');
            expect(consoleErrorSpy).toHaveBeenCalled();
            consoleErrorSpy.mockRestore();
        });

        it('clearUpdates resets all accumulated real-time state', async () => {
            const { result } = renderHook(
                () => useProjectStream('proj-reset', { autoReconnect: false }),
                { wrapper }
            );

            await waitFor(() => {
                expect(result.current.isConnected).toBe(true);
            });

            const ws = MockWebSocket.instances[0];

            act(() => {
                ws.simulateMessage({
                    type: 'paper_discovered',
                    progress: 50,
                    message: 'Discovered paper',
                });
            });

            expect(result.current.progress).toBe(50);
            expect(result.current.totalPapers).toBe(1);

            act(() => {
                result.current.clearUpdates();
            });

            expect(result.current.progress).toBe(0);
            expect(result.current.totalPapers).toBe(0);
            expect(result.current.logs).toEqual([]);
            expect(result.current.updates).toEqual([]);
            expect(result.current.latestCriticVerdict).toBe(null);
            expect(result.current.latestFactCheck).toBe(null);
        });
    });

    // =========================================================================
    // PART 3: COMPONENT RENDERING STRESS & ADVERSARIAL INPUTS
    // =========================================================================
    describe('3. Dashboard Component Rendering Under Stress & Edge-Case Inputs', () => {

        describe('ConflictingDebates component', () => {
            it('renders empty state when debates array is empty or undefined', () => {
                // @ts-expect-error test undefined
                const { rerender } = render(<ConflictingDebates debates={undefined} />);
                expect(screen.getByText('No scientific controversies or debates detected')).toBeInTheDocument();

                rerender(<ConflictingDebates debates={[]} />);
                expect(screen.getByText('No scientific controversies or debates detected')).toBeInTheDocument();
            });

            it('handles search queries with special regex characters without throwing error', () => {
                const sampleDebates: ConflictingDebate[] = [
                    {
                        topic: 'Regex [Test] (Special) * + ? ^ $ \\',
                        perspective_a: 'Perspective A mentions (parentheses) and [brackets].',
                        perspective_b: 'Perspective B discusses * asterisks.',
                        critical_evaluation: 'Evaluation with $ dollar and ^ caret signs.',
                    },
                ];

                render(<ConflictingDebates debates={sampleDebates} />);
                expect(screen.getByText('Regex [Test] (Special) * + ? ^ $ \\')).toBeInTheDocument();

                const searchInput = screen.getByPlaceholderText('Search debates...');
                // Typing regex special character sequence
                fireEvent.change(searchInput, { target: { value: '[Test]' } });
                expect(screen.getByText('Regex [Test] (Special) * + ? ^ $ \\')).toBeInTheDocument();

                fireEvent.change(searchInput, { target: { value: '(*+?)' } });
                expect(screen.queryByText('Regex [Test] (Special) * + ? ^ $ \\')).not.toBeInTheDocument();
            });
        });

        describe('EvidenceMatrixTable component', () => {
            it('handles rows with missing or undefined fields gracefully', () => {
                const partialRows: EvidenceMatrixRow[] = [
                    {
                        paper_id: 'ref_partial',
                        title: 'Partial Data Paper',
                        authors: undefined,
                        year: undefined,
                        methodology: undefined,
                        benchmark_dataset: undefined,
                        primary_metric: undefined,
                        primary_limitation: undefined,
                    },
                ];

                render(<EvidenceMatrixTable rows={partialRows} />);
                expect(screen.getByText('[ref_partial]')).toBeInTheDocument();
                expect(screen.getByText('Partial Data Paper')).toBeInTheDocument();
                expect(screen.getByText('Unknown Authors')).toBeInTheDocument();
            });

            it('handles column sorting and row expansion toggles', () => {
                const rows: EvidenceMatrixRow[] = [
                    {
                        paper_id: 'ref_b',
                        title: 'Beta Architecture',
                        authors: ['Beta Author'],
                        year: 2024,
                        methodology: 'Beta Method',
                        benchmark_dataset: 'Bench B',
                        primary_metric: 'Metric B',
                        primary_limitation: 'Beta Limitation',
                        limitations: ['Beta Limitation 1', 'Beta Limitation 2'],
                        key_findings: ['Beta Finding 1'],
                        is_full_text: true,
                    },
                    {
                        paper_id: 'ref_a',
                        title: 'Alpha Architecture',
                        authors: ['Alpha Author'],
                        year: 2022,
                        methodology: 'Alpha Method',
                        benchmark_dataset: 'Bench A',
                        primary_metric: 'Metric A',
                        primary_limitation: 'Alpha Limitation',
                        is_full_text: false,
                    },
                ];

                render(<EvidenceMatrixTable rows={rows} />);

                // Click title sort header
                const titleHeader = screen.getByRole('button', { name: /Title & Authors/i });
                fireEvent.click(titleHeader);

                // Expand row ref_b
                const betaTitle = screen.getByText('Beta Architecture');
                fireEvent.click(betaTitle);

                expect(screen.getByText('Identified Limitations & Bottlenecks')).toBeInTheDocument();
                expect(screen.getByText('• Beta Limitation 1')).toBeInTheDocument();
                expect(screen.getByText('✓ Beta Finding 1')).toBeInTheDocument();
            });
        });

        describe('ResearchGapViewer component', () => {
            it('handles research gaps with varied grounding structures and priority cases', () => {
                const gaps: ResearchGapItem[] = [
                    {
                        gap_id: 'GAP-HIGH',
                        title: 'High Priority Gap',
                        description: 'Substantive high priority gap description.',
                        importance: 'high',
                        recommended_methodology: 'Methodology for high priority gap.',
                        grounding_paper_ids: ['ref_1', 'ref_2'],
                    },
                    {
                        gap_id: 'GAP-MED',
                        title: 'Medium Priority Gap',
                        description: 'Medium priority description.',
                        importance: 'medium',
                        recommended_methodology: 'Medium roadmap.',
                        // Testing object-style grounding structure
                        grounding_papers: [{ paper_id: 'ref_3', title: 'Paper 3' }] as unknown as string[],
                    },
                ];

                render(<ResearchGapViewer gaps={gaps} />);
                expect(screen.getByText('HIGH PRIORITY')).toBeInTheDocument();
                expect(screen.getByText('MEDIUM PRIORITY')).toBeInTheDocument();
                expect(screen.getByText('[ref_1]')).toBeInTheDocument();
                expect(screen.getByText('[ref_3]')).toBeInTheDocument();

                // Test copy summary button interaction
                const writeTextMock = vi.fn();
                Object.assign(navigator, {
                    clipboard: { writeText: writeTextMock },
                });

                const copyButtons = screen.getAllByRole('button', { name: /Copy Summary/i });
                fireEvent.click(copyButtons[0]);
                expect(writeTextMock).toHaveBeenCalledWith(expect.stringContaining('High Priority Gap'));
            });
        });

        describe('MethodologyDistributionCard component', () => {
            it('returns null if overview is undefined and renders empty fallback if distribution is empty', () => {
                const { container, rerender } = render(<MethodologyDistributionCard overview={undefined} />);
                expect(container.firstChild).toBeNull();

                const emptyOverview: MethodologyDistribution = {
                    distribution: {},
                    dominant_approach: '',
                    trend_description: '',
                };

                rerender(<MethodologyDistributionCard overview={emptyOverview} />);
                expect(screen.getByText('Methodological Landscape & Distribution')).toBeInTheDocument();
                expect(screen.getByText('Methodology distribution data will be computed as papers are analyzed.')).toBeInTheDocument();
            });

            it('calculates percentage bars correctly across multi-method distributions', () => {
                const overview: MethodologyDistribution = {
                    dominant_approach: 'Sparse Attention',
                    trend_description: 'Rapid growth in memory-efficient sparsification.',
                    distribution: {
                        'Sparse Attention': 6,
                        'Dense Attention': 2,
                        'Linear Recurrence': 2,
                    },
                };

                render(<MethodologyDistributionCard overview={overview} totalPapers={10} />);
                expect(screen.getAllByText('Sparse Attention').length).toBeGreaterThanOrEqual(1);
                expect(screen.getByText('6 papers (60%)')).toBeInTheDocument();
                expect(screen.getAllByText('2 papers (20%)').length).toBe(2);
                expect(screen.getByText('Rapid growth in memory-efficient sparsification.')).toBeInTheDocument();
            });
        });

        describe('ThematicSections component & interactive citation resolution', () => {
            it('renders markdown prose with clickable citation chips and resolves citation popover data', async () => {
                const sections: ThematicSection[] = [
                    {
                        theme_id: 'sec_1',
                        title: 'Memory Optimization Strategies',
                        synthesis_prose: 'Linear attention reduces complexity to O(N) as established in [ref_flash#sec_io] and validated in [ref_2]. Block sparse routing is also viable [ref_missing].',
                        key_takeaways: ['O(N) memory complexity achieved.', 'Hardware-aware tiling is optimal.'],
                        cited_paper_ids: ['ref_flash', 'ref_2', 'ref_missing'],
                    },
                ];

                const bibliography: BibliographyItem[] = [
                    {
                        paper_id: 'ref_flash',
                        title: 'FlashAttention: Fast and Memory-Efficient Exact Attention',
                        authors: ['Tri Dao', 'Daniel Fu'],
                        year: 2022,
                        doi: '10.48550/arXiv.2205.14135',
                        pdf_url: 'https://arxiv.org/pdf/2205.14135.pdf',
                    },
                ];

                render(<ThematicSections sections={sections} bibliography={bibliography} />);

                expect(screen.getByText('Memory Optimization Strategies')).toBeInTheDocument();
                expect(screen.getByText('Key Findings & Actionable Takeaways')).toBeInTheDocument();

                // Click citation chip [ref_flash#sec_io]
                const flashChip = screen.getByText('[ref_flash#sec_io]');
                expect(flashChip).toBeInTheDocument();
                fireEvent.click(flashChip);

                await waitFor(() => {
                    expect(screen.getByText(/Citation \[ref_flash#sec_io\]/i)).toBeInTheDocument();
                    expect(screen.getByText('FlashAttention: Fast and Memory-Efficient Exact Attention')).toBeInTheDocument();
                    expect(screen.getByText(/Tri Dao and Daniel Fu \(2022\)/i)).toBeInTheDocument();
                    expect(screen.getByText('§ sec io')).toBeInTheDocument();
                    expect(screen.getByText('View Source')).toBeInTheDocument();
                });

                // Close popover
                const closeBtn = screen.getByTestId('CloseIcon');
                fireEvent.click(closeBtn);

                // Click uncited / missing paper chip [ref_missing]
                const missingChip = screen.getByText('[ref_missing]');
                fireEvent.click(missingChip);

                await waitFor(() => {
                    expect(screen.getByText(/Grounding evidence anchored to/i)).toBeInTheDocument();
                    expect(screen.getByText('ref_missing')).toBeInTheDocument();
                });
            });
        });
    });
});
