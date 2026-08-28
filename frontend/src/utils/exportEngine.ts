import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import {
    Document,
    Packer,
    Paragraph,
    TextRun,
    HeadingLevel,
    Table,
    TableRow,
    TableCell,
    WidthType,
    ShadingType,
    AlignmentType,
} from 'docx';
import { saveAs } from 'file-saver';
import type {
    ResearchReport,
    BibliographyItem,
    EvidenceMatrixRow,
    PaperReference,
} from '../types';

/**
 * Cleanly format paper authors into standard academic citation string
 */
export function formatAuthors(authors?: string[] | null): string {
    if (!authors || !Array.isArray(authors) || authors.length === 0) return 'Unknown Authors';
    const cleanList = authors
        .map((a) => (typeof a === 'string' ? a.trim() : ''))
        .filter((a) => a.length > 0 && a.toLowerCase() !== 'unknown' && a.toLowerCase() !== 'unknown authors');
    if (cleanList.length === 0) return 'Unknown Authors';
    if (cleanList.length === 1) return cleanList[0];
    if (cleanList.length === 2) return `${cleanList[0]} and ${cleanList[1]}`;
    return `${cleanList[0]} et al.`;
}

/**
 * Generate a clean, informative, slugified filename for exported deliverables
 */
export function slugifyTitle(title?: string): string {
    if (!title || !title.trim()) return 'literature_review';
    const clean = title
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '');
    return clean.substring(0, 45) || 'literature_review';
}

export function generateExportFilename(title?: string, format: string = 'pdf'): string {
    const slug = slugifyTitle(title);
    const date = new Date().toISOString().slice(0, 10);
    return `${slug}_${date}.${format}`;
}

/**
 * Generate standard Markdown formatted text from ResearchReport deliverable
 */
export function generateReportMarkdown(report: ResearchReport): string {
    const mdLines: string[] = [
        `# ${report.metadata?.title || report.title || 'Scientific Literature Review'}`,
        '',
        `**Research Question:** ${report.metadata?.research_question || 'N/A'}  `,
        `**Generated:** ${report.metadata?.generated_at || new Date().toISOString()}  `,
        `**Quality Score:** ${report.metadata?.quality_score !== undefined ? report.metadata.quality_score.toFixed(1) : (report.quality_score?.toFixed(1) ?? 'N/A')}/100 | **Status:** ${(report.metadata?.status || 'COMPLETE').toString().toUpperCase()}`,
        '',
        '---',
        '',
        '## Executive Summary',
        '',
        report.executive_summary || 'No executive summary generated.',
        '',
    ];

    // Evidence Comparison Matrix
    const matrixRows = report.comparison_matrix || report.comparative_matrix || [];
    if (matrixRows.length > 0) {
        mdLines.push('## Evidence Comparison Matrix');
        mdLines.push('');
        mdLines.push('| Paper ID | Title | Authors | Year | Methodology | Benchmark Dataset | Primary Metric | Primary Limitation | Full-Text |');
        mdLines.push('| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |');

        const esc = (s?: string) => (s ? String(s).replace(/\|/g, '\\|').replace(/\n/g, ' ') : '-');

        for (const row of matrixRows) {
            const authStr = formatAuthors(row.authors);
            const yrStr = row.year ? String(row.year) : 'N/A';
            const ftStr = row.is_full_text || row.has_full_text ? 'Yes (Full-Text)' : 'No (Abstract)';
            mdLines.push(
                `| ${esc(row.paper_id)} | ${esc(row.title)} | ${esc(authStr)} | ${yrStr} | ${esc(row.methodology || row.methodology_type)} | ${esc(row.benchmark_dataset || row.dataset)} | ${esc(row.primary_metric)} | ${esc(row.primary_limitation || (row.limitations?.[0]))} | ${ftStr} |`
            );
        }
        mdLines.push('');
    }

    // Methodology Overview
    if (report.methodology_overview) {
        mdLines.push('## Methodology Overview');
        mdLines.push('');
        mdLines.push(`**Dominant Approach:** ${report.methodology_overview.dominant_approach || 'N/A'}`);
        mdLines.push('');
        if (report.methodology_overview.trend_description) {
            mdLines.push(report.methodology_overview.trend_description);
            mdLines.push('');
        }
        if (report.methodology_overview.distribution && Object.keys(report.methodology_overview.distribution).length > 0) {
            mdLines.push('### Distribution Breakdown');
            for (const [approach, count] of Object.entries(report.methodology_overview.distribution)) {
                mdLines.push(`- **${approach}**: ${count} papers`);
            }
            mdLines.push('');
        }
    }

    // Thematic Sections
    const sections = report.thematic_sections || report.sections || [];
    if (sections.length > 0) {
        mdLines.push('## Thematic Synthesis');
        mdLines.push('');
        for (const section of sections) {
            mdLines.push(`### ${section.title}`);
            mdLines.push('');
            mdLines.push(section.synthesis_prose || section.content || '');
            mdLines.push('');
            if (section.key_takeaways && section.key_takeaways.length > 0) {
                mdLines.push('**Key Takeaways:**');
                for (const item of section.key_takeaways) {
                    mdLines.push(`- ${item}`);
                }
                mdLines.push('');
            }
        }
    }

    // Conflicting Debates
    const debates = report.conflicting_findings_and_debates || report.conflicting_debates || report.debates || [];
    if (debates.length > 0) {
        mdLines.push('## Conflicting Findings & Scientific Debates');
        mdLines.push('');
        for (const debate of debates) {
            mdLines.push(`### Debate: ${debate.topic}`);
            mdLines.push('');
            mdLines.push(`**Perspective A:** ${debate.perspective_a}`);
            mdLines.push('');
            mdLines.push(`**Perspective B:** ${debate.perspective_b}`);
            mdLines.push('');
            mdLines.push(`**Critical Evaluation:** ${debate.critical_evaluation}`);
            mdLines.push('');
        }
    }

    // Actionable Research Gaps
    const gaps = report.actionable_research_gaps || report.actionable_gaps || report.research_gaps || [];
    if (gaps.length > 0) {
        mdLines.push('## Actionable Research Gaps & Future Directions');
        mdLines.push('');
        for (const gap of gaps) {
            const priority = (gap.importance || gap.priority || 'high').toUpperCase();
            const grounding = (gap.grounding_paper_ids || gap.grounding_papers || []).join(', ') || 'Corpus';
            mdLines.push(`### [${priority} PRIORITY] ${gap.gap_id || gap.id || 'Gap'}: ${gap.title || gap.description}`);
            mdLines.push('');
            if (gap.title && gap.description && gap.title !== gap.description) {
                mdLines.push(gap.description);
                mdLines.push('');
            }
            mdLines.push(`- **Grounding Papers:** ${grounding}`);
            mdLines.push(`- **Recommended Methodology:** ${gap.recommended_methodology}`);
            mdLines.push('');
        }
    }

    // Bibliography
    const bibItems = report.bibliography || [];
    if (bibItems.length > 0) {
        mdLines.push('## Bibliography');
        mdLines.push('');
        for (const item of bibItems) {
            const authStr = formatAuthors(item.authors);
            const yrStr = item.year ? `(${item.year})` : '';
            const venueStr = item.venue || item.source ? `*${item.venue || item.source}*.` : '';
            const doiStr = item.doi ? `DOI: [${item.doi}](https://doi.org/${item.doi})` : '';
            const urlStr = item.pdf_url || item.url ? `[PDF/Link](${item.pdf_url || item.url})` : '';
            const ftStr = item.is_full_text_analyzed ? '[Full-Text Analyzed]' : '[Abstract Only]';
            const links = [doiStr, urlStr, ftStr].filter(Boolean).join(' | ');
            mdLines.push(`- **[${item.paper_id}]** ${authStr} ${yrStr}. **${item.title}**. ${venueStr} ${links}`);
        }
        mdLines.push('');
    }

    return mdLines.join('\n');
}

/**
 * Export report to Markdown (.md) file
 */
export function exportToMarkdown(report: ResearchReport, filename: string = 'literature-review.md'): void {
    const mdContent = generateReportMarkdown(report);
    const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8' });
    saveAs(blob, filename);
}

/**
 * Generate standard BibTeX string from paper items
 */
export function generateBibTeX(items: (BibliographyItem | EvidenceMatrixRow | PaperReference)[]): string {
    const entries: string[] = [];

    for (const item of items) {
        const paperId = (item as BibliographyItem).paper_id || (item as PaperReference).id || 'ref';
        const cleanKey = paperId.replace(/[^a-zA-Z0-9_-]/g, '_');
        const title = item.title || 'Untitled';
        const authors = (item.authors || []).join(' and ') || 'Unknown';
        const year = item.year || new Date().getFullYear();
        const venue = (item as BibliographyItem).venue || 'arXiv preprint';
        const doi = (item as BibliographyItem).doi || '';
        const url = (item as BibliographyItem).pdf_url || (item as PaperReference).url || '';

        const lines = [
            `@article{${cleanKey},`,
            `  title={${title}},`,
            `  author={${authors}},`,
            `  year={${year}},`,
            `  journal={${venue}},`,
        ];

        if (doi) lines.push(`  doi={${doi}},`);
        if (url) lines.push(`  url={${url}},`);
        lines.push('}');

        entries.push(lines.join('\n'));
    }

    return entries.join('\n\n');
}

/**
 * Export bibliography to BibTeX (.bib) file
 */
export function exportToBibTeX(
    items: (BibliographyItem | EvidenceMatrixRow | PaperReference)[],
    filename: string = 'references.bib'
): void {
    const bibContent = generateBibTeX(items);
    const blob = new Blob([bibContent], { type: 'application/x-bibtex;charset=utf-8' });
    saveAs(blob, filename);
}

/**
 * Export full ResearchReport to professional Word Document (.docx)
 */
export async function exportToDocx(
    report: ResearchReport,
    filename: string = 'literature-review.docx'
): Promise<void> {
    const children: (Paragraph | Table)[] = [];

    // Title
    children.push(
        new Paragraph({
            text: report.metadata?.title || report.title || 'Autonomous Scientific Literature Review',
            heading: HeadingLevel.TITLE,
            alignment: AlignmentType.CENTER,
            spacing: { after: 200 },
        })
    );

    // Subtitle / Metadata callout
    children.push(
        new Paragraph({
            children: [
                new TextRun({ text: 'Research Question: ', bold: true }),
                new TextRun({ text: report.metadata?.research_question || 'N/A', italics: true }),
            ],
            spacing: { after: 120 },
        }),
        new Paragraph({
            children: [
                new TextRun({ text: `Generated: ${report.metadata?.generated_at || new Date().toISOString()}  |  `, color: '666666' }),
                new TextRun({ text: `Quality Score: ${report.metadata?.quality_score !== undefined ? report.metadata.quality_score.toFixed(1) : (report.quality_score?.toFixed(1) ?? 'N/A')}/100`, bold: true, color: '008060' }),
            ],
            spacing: { after: 300 },
        })
    );

    // Executive Summary
    children.push(
        new Paragraph({
            text: 'Executive Summary',
            heading: HeadingLevel.HEADING_1,
            spacing: { before: 240, after: 120 },
        }),
        new Paragraph({
            children: [new TextRun({ text: report.executive_summary || 'No executive summary provided.' })],
            spacing: { after: 240 },
        })
    );

    // Evidence Matrix Table
    const matrixRows = report.comparison_matrix || report.comparative_matrix || [];
    if (matrixRows.length > 0) {
        children.push(
            new Paragraph({
                text: 'Evidence Comparison Matrix',
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 240, after: 120 },
            })
        );

        const tableHeader = new TableRow({
            children: [
                new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Paper ID', bold: true, color: 'FFFFFF' })] })], shading: { type: ShadingType.CLEAR, fill: '1E293B' }, width: { size: 12, type: WidthType.PERCENTAGE } }),
                new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Title & Authors', bold: true, color: 'FFFFFF' })] })], shading: { type: ShadingType.CLEAR, fill: '1E293B' }, width: { size: 28, type: WidthType.PERCENTAGE } }),
                new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Methodology', bold: true, color: 'FFFFFF' })] })], shading: { type: ShadingType.CLEAR, fill: '1E293B' }, width: { size: 20, type: WidthType.PERCENTAGE } }),
                new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Benchmarks & Metric', bold: true, color: 'FFFFFF' })] })], shading: { type: ShadingType.CLEAR, fill: '1E293B' }, width: { size: 20, type: WidthType.PERCENTAGE } }),
                new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Key Limitations', bold: true, color: 'FFFFFF' })] })], shading: { type: ShadingType.CLEAR, fill: '1E293B' }, width: { size: 20, type: WidthType.PERCENTAGE } }),
            ],
        });

        const tableDataRows = matrixRows.map((row, idx) => {
            const fill = idx % 2 === 0 ? 'F8FAFC' : 'FFFFFF';
            const authYear = `${formatAuthors(row.authors)} (${row.year || 'N/A'})`;
            const benchMetric = `${row.benchmark_dataset || row.dataset || 'Benchmark'}\nMetric: ${row.primary_metric || '-'}`;
            return new TableRow({
                children: [
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: row.paper_id, bold: true })] })], shading: { type: ShadingType.CLEAR, fill } }),
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: row.title, bold: true })] }), new Paragraph({ children: [new TextRun({ text: authYear, italics: true, color: '555555' })] })], shading: { type: ShadingType.CLEAR, fill } }),
                    new TableCell({ children: [new Paragraph({ text: row.methodology || row.methodology_type || '-' })], shading: { type: ShadingType.CLEAR, fill } }),
                    new TableCell({ children: [new Paragraph({ text: benchMetric })], shading: { type: ShadingType.CLEAR, fill } }),
                    new TableCell({ children: [new Paragraph({ text: row.primary_limitation || (row.limitations?.[0] || '-') })], shading: { type: ShadingType.CLEAR, fill } }),
                ],
            });
        });

        children.push(
            new Table({
                rows: [tableHeader, ...tableDataRows],
                width: { size: 100, type: WidthType.PERCENTAGE },
            })
        );
        children.push(new Paragraph({ text: '', spacing: { after: 200 } }));
    }

    // Thematic Sections
    const sections = report.thematic_sections || report.sections || [];
    if (sections.length > 0) {
        children.push(
            new Paragraph({
                text: 'Thematic Synthesis',
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 240, after: 120 },
            })
        );

        for (const sec of sections) {
            children.push(
                new Paragraph({
                    text: sec.title,
                    heading: HeadingLevel.HEADING_2,
                    spacing: { before: 180, after: 80 },
                }),
                new Paragraph({
                    text: sec.synthesis_prose || sec.content || '',
                    spacing: { after: 120 },
                })
            );

            if (sec.key_takeaways && sec.key_takeaways.length > 0) {
                children.push(
                    new Paragraph({
                        children: [new TextRun({ text: 'Key Takeaways:', bold: true })],
                        spacing: { before: 60, after: 40 },
                    })
                );
                for (const t of sec.key_takeaways) {
                    children.push(
                        new Paragraph({
                            text: t,
                            bullet: { level: 0 },
                            spacing: { after: 40 },
                        })
                    );
                }
            }
        }
    }

    // Conflicting Debates
    const debates = report.conflicting_findings_and_debates || report.conflicting_debates || report.debates || [];
    if (debates.length > 0) {
        children.push(
            new Paragraph({
                text: 'Methodological & Scientific Debates',
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 240, after: 120 },
            })
        );

        for (const debate of debates) {
            children.push(
                new Paragraph({
                    text: `Debate: ${debate.topic}`,
                    heading: HeadingLevel.HEADING_2,
                    spacing: { before: 160, after: 80 },
                }),
                new Paragraph({
                    children: [
                        new TextRun({ text: 'Perspective A: ', bold: true, color: '2563EB' }),
                        new TextRun({ text: debate.perspective_a }),
                    ],
                    spacing: { after: 80 },
                }),
                new Paragraph({
                    children: [
                        new TextRun({ text: 'Perspective B: ', bold: true, color: 'DC2626' }),
                        new TextRun({ text: debate.perspective_b }),
                    ],
                    spacing: { after: 80 },
                }),
                new Paragraph({
                    children: [
                        new TextRun({ text: 'Critical Evaluation: ', bold: true, color: '059669' }),
                        new TextRun({ text: debate.critical_evaluation, italics: true }),
                    ],
                    spacing: { after: 160 },
                })
            );
        }
    }

    // Actionable Research Gaps
    const gaps = report.actionable_research_gaps || report.actionable_gaps || report.research_gaps || [];
    if (gaps.length > 0) {
        children.push(
            new Paragraph({
                text: 'Actionable Research Gaps & Future Work',
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 240, after: 120 },
            })
        );

        for (const gap of gaps) {
            const priority = (gap.importance || gap.priority || 'high').toUpperCase();
            children.push(
                new Paragraph({
                    children: [
                        new TextRun({ text: `[${priority} PRIORITY] `, bold: true, color: priority === 'HIGH' ? 'DC2626' : 'D97706' }),
                        new TextRun({ text: `${gap.gap_id || 'Gap'}: ${gap.title || gap.description}`, bold: true }),
                    ],
                    spacing: { before: 120, after: 60 },
                }),
                new Paragraph({
                    children: [
                        new TextRun({ text: 'Recommended Methodology: ', bold: true }),
                        new TextRun({ text: gap.recommended_methodology }),
                    ],
                    spacing: { after: 60 },
                }),
                new Paragraph({
                    children: [
                        new TextRun({ text: 'Grounding Papers: ', bold: true, color: '666666' }),
                        new TextRun({ text: (gap.grounding_paper_ids || gap.grounding_papers || []).join(', ') || 'Corpus' }),
                    ],
                    spacing: { after: 140 },
                })
            );
        }
    }

    // Bibliography
    const bibliography = report.bibliography || [];
    if (bibliography.length > 0) {
        children.push(
            new Paragraph({
                text: 'References & Bibliography',
                heading: HeadingLevel.HEADING_1,
                spacing: { before: 240, after: 120 },
            })
        );

        for (const item of bibliography) {
            const authStr = formatAuthors(item.authors);
            const yrStr = item.year ? `(${item.year})` : '';
            const venueStr = item.venue || item.source ? `${item.venue || item.source}.` : '';
            children.push(
                new Paragraph({
                    children: [
                        new TextRun({ text: `[${item.paper_id}] `, bold: true }),
                        new TextRun({ text: `${authStr} ${yrStr}. ` }),
                        new TextRun({ text: `"${item.title}". `, bold: true }),
                        new TextRun({ text: venueStr, italics: true }),
                        item.doi ? new TextRun({ text: ` DOI: ${item.doi}`, color: '2563EB' }) : new TextRun({ text: '' }),
                    ],
                    spacing: { after: 80 },
                })
            );
        }
    }

    const doc = new Document({
        sections: [
            {
                properties: {},
                children,
            },
        ],
    });

    const blob = await Packer.toBlob(doc);
    saveAs(blob, filename);
}

/**
 * Generate a complete, academic light-themed printable HTML DOM for PDF rendering
 */
export function createAcademicPrintableContainer(report: ResearchReport): HTMLDivElement {
    const container = document.createElement('div');
    container.id = 'academic-pdf-export-root';
    container.style.cssText = `
        background-color: #FFFFFF;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 13px;
        line-height: 1.6;
        padding: 40px;
        width: 800px;
        box-sizing: border-box;
    `;

    const title = report.metadata?.title || report.title || 'Autonomous Scientific Literature Review';
    const question = report.metadata?.research_question || 'Literature Synthesis';
    const qualityScore = report.metadata?.quality_score !== undefined
        ? report.metadata.quality_score.toFixed(1)
        : (report.quality_score?.toFixed(1) ?? '85.0');
    const generatedDate = report.metadata?.generated_at
        ? new Date(report.metadata.generated_at).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
        : new Date().toLocaleDateString();

    const matrixRows = report.comparison_matrix || report.comparative_matrix || [];
    const sections = report.thematic_sections || report.sections || [];
    const debates = report.conflicting_findings_and_debates || report.conflicting_debates || report.debates || [];
    const gaps = report.actionable_research_gaps || report.actionable_gaps || report.research_gaps || [];
    const bibliography = report.bibliography || [];

    let matrixHtml = '';
    if (matrixRows.length > 0) {
        const rowsHtml = matrixRows.map((r, i) => `
            <tr style="background-color: ${i % 2 === 0 ? '#F8FAFC' : '#FFFFFF'}; break-inside: avoid; page-break-inside: avoid;">
                <td style="padding: 8px 10px; border: 1px solid #CBD5E1; font-weight: 700; color: #D97706; font-family: monospace; font-size: 11px;">[${r.paper_id}]</td>
                <td style="padding: 8px 10px; border: 1px solid #CBD5E1;">
                    <div style="font-weight: 700; color: #0F172A; font-size: 12px; margin-bottom: 2px;">${r.title}</div>
                    <div style="color: #64748B; font-size: 11px; font-style: italic;">${formatAuthors(r.authors)} ${r.year ? `(${r.year})` : ''}</div>
                </td>
                <td style="padding: 8px 10px; border: 1px solid #CBD5E1; color: #0F172A; font-size: 11px;">${r.methodology || r.methodology_type || '-'}</td>
                <td style="padding: 8px 10px; border: 1px solid #CBD5E1; color: #0F172A; font-size: 11px;">
                    <div>${r.benchmark_dataset || r.dataset || 'Standard Corpus'}</div>
                    <div style="color: #0284C7; font-weight: 600; margin-top: 2px;">${r.primary_metric || '-'}</div>
                </td>
                <td style="padding: 8px 10px; border: 1px solid #CBD5E1; color: #334155; font-size: 11px;">${r.primary_limitation || (r.limitations?.[0] || '-')}</td>
            </tr>
        `).join('');

        matrixHtml = `
            <div style="margin-top: 30px; margin-bottom: 24px; break-inside: avoid; page-break-inside: avoid;">
                <h2 style="font-size: 18px; font-weight: 800; color: #0F172A; border-bottom: 2px solid #0F172A; padding-bottom: 6px; margin-bottom: 12px;">Evidence Comparison Matrix</h2>
                <table style="width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 11px;">
                    <thead>
                        <tr style="background-color: #0F172A; color: #FFFFFF;">
                            <th style="padding: 8px 10px; border: 1px solid #0F172A; text-align: left; width: 10%;">Ref ID</th>
                            <th style="padding: 8px 10px; border: 1px solid #0F172A; text-align: left; width: 32%;">Title & Authors</th>
                            <th style="padding: 8px 10px; border: 1px solid #0F172A; text-align: left; width: 18%;">Methodology</th>
                            <th style="padding: 8px 10px; border: 1px solid #0F172A; text-align: left; width: 20%;">Benchmark & Metric</th>
                            <th style="padding: 8px 10px; border: 1px solid #0F172A; text-align: left; width: 20%;">Primary Limitation</th>
                        </tr>
                    </thead>
                    <tbody>${rowsHtml}</tbody>
                </table>
            </div>
        `;
    }

    let thematicHtml = '';
    if (sections.length > 0) {
        thematicHtml = `
            <div style="margin-top: 30px; margin-bottom: 24px;">
                <h2 style="font-size: 18px; font-weight: 800; color: #0F172A; border-bottom: 2px solid #0F172A; padding-bottom: 6px; margin-bottom: 16px;">Thematic Synthesis</h2>
                ${sections.map((sec) => `
                    <div style="margin-bottom: 20px; break-inside: avoid; page-break-inside: avoid;">
                        <h3 style="font-size: 15px; font-weight: 700; color: #1E293B; margin-bottom: 8px;">${sec.title}</h3>
                        <p style="color: #334155; font-size: 13px; line-height: 1.7; margin-bottom: 10px;">${sec.synthesis_prose || sec.content || ''}</p>
                        ${sec.key_takeaways && sec.key_takeaways.length > 0 ? `
                            <div style="background-color: #F8FAFC; border-left: 3px solid #3B82F6; padding: 10px 14px; border-radius: 4px; margin-top: 8px;">
                                <strong style="font-size: 12px; color: #1E293B;">Key Takeaways:</strong>
                                <ul style="margin: 4px 0 0 16px; padding: 0; color: #334155; font-size: 12px;">
                                    ${sec.key_takeaways.map((t) => `<li style="margin-bottom: 3px;">${t}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }

    let debatesHtml = '';
    if (debates.length > 0) {
        debatesHtml = `
            <div style="margin-top: 30px; margin-bottom: 24px;">
                <h2 style="font-size: 18px; font-weight: 800; color: #0F172A; border-bottom: 2px solid #0F172A; padding-bottom: 6px; margin-bottom: 16px;">Scientific & Methodological Debates</h2>
                ${debates.map((d) => `
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; break-inside: avoid; page-break-inside: avoid;">
                        <h3 style="font-size: 14px; font-weight: 700; color: #0F172A; margin: 0 0 8px 0;">Debate: ${d.topic}</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px;">
                            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 10px;">
                                <strong style="color: #1D4ED8; font-size: 11px; text-transform: uppercase;">Perspective A:</strong>
                                <p style="margin: 4px 0 0 0; font-size: 12px; color: #1E293B;">${d.perspective_a}</p>
                            </div>
                            <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 10px;">
                                <strong style="color: #B91C1C; font-size: 11px; text-transform: uppercase;">Perspective B:</strong>
                                <p style="margin: 4px 0 0 0; font-size: 12px; color: #1E293B;">${d.perspective_b}</p>
                            </div>
                        </div>
                        <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 6px; padding: 10px;">
                            <strong style="color: #047857; font-size: 11px; text-transform: uppercase;">Critical Evaluation:</strong>
                            <p style="margin: 4px 0 0 0; font-size: 12px; color: #064E3B; font-style: italic;">${d.critical_evaluation}</p>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    let gapsHtml = '';
    if (gaps.length > 0) {
        gapsHtml = `
            <div style="margin-top: 30px; margin-bottom: 24px;">
                <h2 style="font-size: 18px; font-weight: 800; color: #0F172A; border-bottom: 2px solid #0F172A; padding-bottom: 6px; margin-bottom: 16px;">Actionable Research Gaps & Future Directions</h2>
                ${gaps.map((g) => {
                    const prio = (g.importance || g.priority || 'high').toUpperCase();
                    const prioColor = prio === 'HIGH' ? '#B91C1C' : '#D97706';
                    const prioBg = prio === 'HIGH' ? '#FEF2F2' : '#FFFBEB';
                    const grounding = (g.grounding_paper_ids || g.grounding_papers || []).join(', ') || 'Corpus';
                    return `
                        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid ${prioColor}; border-radius: 8px; padding: 14px 18px; margin-bottom: 14px; break-inside: avoid; page-break-inside: avoid;">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                                <span style="background-color: ${prioBg}; color: ${prioColor}; font-weight: 800; font-size: 10px; padding: 2px 8px; border-radius: 4px; border: 1px solid ${prioColor}40;">${prio} PRIORITY</span>
                                <span style="color: #64748B; font-family: monospace; font-size: 11px; font-weight: 700;">${g.gap_id || 'GAP'}</span>
                            </div>
                            <p style="color: #0F172A; font-size: 13px; font-weight: 500; margin: 4px 0 8px 0;">${g.description}</p>
                            <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 8px 12px; margin-bottom: 6px;">
                                <strong style="color: #15803D; font-size: 11px; text-transform: uppercase;">Actionable Roadmap:</strong>
                                <span style="color: #166534; font-size: 12px;"> ${g.recommended_methodology}</span>
                            </div>
                            <div style="color: #64748B; font-size: 11px;">
                                <strong>Grounding Literature:</strong> ${grounding}
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    let bibHtml = '';
    if (bibliography.length > 0) {
        bibHtml = `
            <div style="margin-top: 30px; margin-bottom: 24px; break-inside: avoid; page-break-inside: avoid;">
                <h2 style="font-size: 18px; font-weight: 800; color: #0F172A; border-bottom: 2px solid #0F172A; padding-bottom: 6px; margin-bottom: 16px;">References & Bibliography</h2>
                <ol style="padding-left: 20px; margin: 0; color: #334155; font-size: 12px;">
                    ${bibliography.map((b) => {
                        const authStr = formatAuthors(b.authors);
                        const yrStr = b.year ? `(${b.year})` : '';
                        const venueStr = b.venue || b.source ? `<em>${b.venue || b.source}</em>.` : '';
                        const doiStr = b.doi ? `<span style="color: #2563EB;">DOI: ${b.doi}</span>` : '';
                        return `
                            <li style="margin-bottom: 8px; line-height: 1.5; break-inside: avoid; page-break-inside: avoid;">
                                <strong>[${b.paper_id}]</strong> ${authStr} ${yrStr}. <strong>"${b.title}"</strong>. ${venueStr} ${doiStr}
                            </li>
                        `;
                    }).join('')}
                </ol>
            </div>
        `;
    }

    container.innerHTML = `
        <div style="border-bottom: 3px solid #0F172A; padding-bottom: 16px; margin-bottom: 24px;">
            <div style="font-size: 11px; font-weight: 800; letter-spacing: 0.1em; color: #D97706; text-transform: uppercase; margin-bottom: 4px;">Scholar Agent — Autonomous Scientific Synthesis</div>
            <h1 style="font-size: 24px; font-weight: 900; color: #0F172A; line-height: 1.25; margin: 0 0 10px 0;">${title}</h1>
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="font-size: 12px; color: #64748B; font-weight: 700; text-transform: uppercase; margin-bottom: 2px;">Target Research Question</div>
                <div style="font-size: 14px; color: #0F172A; font-style: italic; font-weight: 500;">"${question}"</div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; color: #64748B;">
                <div>Generated: <strong>${generatedDate}</strong></div>
                <div style="color: #059669; font-weight: 700;">Critic Quality Score: ${qualityScore}/100</div>
            </div>
        </div>

        <div style="margin-bottom: 24px; break-inside: avoid; page-break-inside: avoid;">
            <h2 style="font-size: 18px; font-weight: 800; color: #0F172A; border-bottom: 2px solid #0F172A; padding-bottom: 6px; margin-bottom: 12px;">Executive Summary</h2>
            <p style="color: #1E293B; font-size: 13px; line-height: 1.8; margin: 0;">${report.executive_summary || 'Autonomous literature synthesis in progress.'}</p>
        </div>

        ${matrixHtml}
        ${thematicHtml}
        ${debatesHtml}
        ${gapsHtml}
        ${bibHtml}

        <div style="margin-top: 40px; padding-top: 12px; border-top: 1px solid #E2E8F0; display: flex; justify-content: space-between; font-size: 10px; color: #94A3B8;">
            <span>Scholar Agent Autonomous Multi-Agent Synthesis Engine</span>
            <span>Generated from peer-reviewed scientific literature</span>
        </div>
    `;

    return container;
}

/**
 * Export rendered ResearchReport to professional academic PDF (.pdf) with multi-page support
 */
export async function exportToPdf(
    reportOrElementId: ResearchReport | string = 'synthesis-output',
    filename?: string
): Promise<void> {
    let report: ResearchReport | null = null;
    let targetFilename = filename;

    if (typeof reportOrElementId !== 'string' && reportOrElementId && typeof reportOrElementId === 'object') {
        report = reportOrElementId;
        if (!targetFilename) {
            targetFilename = generateExportFilename(report.metadata?.title || report.title, 'pdf');
        }
    } else if (typeof reportOrElementId === 'string') {
        targetFilename = targetFilename || 'literature-review.pdf';
    }

    if (!targetFilename) {
        targetFilename = 'literature-review.pdf';
    }

    // Build dedicated academic light container
    let container: HTMLDivElement | null = null;
    if (report) {
        container = createAcademicPrintableContainer(report);
    } else {
        const el = document.getElementById(reportOrElementId as string);
        if (el) {
            container = el.cloneNode(true) as HTMLDivElement;
            container.style.backgroundColor = '#FFFFFF';
            container.style.color = '#0F172A';
        }
    }

    if (!container) {
        throw new Error('No valid printable container or element found for PDF export.');
    }

    // CRITICAL: Position container in active coordinate space (top=0, left=0) with zIndex=-99999
    // Do NOT use negative coordinates (e.g. left: -10000px) as html2canvas clips to document viewport!
    container.style.position = 'fixed';
    container.style.top = '0px';
    container.style.left = '0px';
    container.style.width = '800px';
    container.style.maxWidth = '800px';
    container.style.zIndex = '-99999';
    container.style.opacity = '1';
    container.style.visibility = 'visible';
    container.style.pointerEvents = 'none';
    document.body.appendChild(container);

    try {
        // Allow browser to calculate styles and layout
        await new Promise((resolve) => setTimeout(resolve, 150));

        const canvas = await html2canvas(container, {
            scale: 2, // High resolution (300 DPI equivalent)
            useCORS: true,
            allowTaint: true,
            backgroundColor: '#FFFFFF',
            width: 800,
            windowWidth: 800,
            scrollX: 0,
            scrollY: 0,
            logging: false,
        });

        if (!canvas || canvas.width === 0 || canvas.height === 0) {
            throw new Error('Canvas rendering produced empty image buffer.');
        }

        const pdf = new jsPDF({
            orientation: 'portrait',
            unit: 'mm',
            format: 'a4',
        });

        const pdfWidth = 210; // A4 mm width
        const pdfHeight = 297; // A4 mm height
        const margin = 10; // 10mm margins
        const contentWidth = pdfWidth - 2 * margin; // 190mm
        const contentHeight = pdfHeight - 2 * margin; // 277mm

        // Height of 1 PDF page slice in canvas pixel coordinate system
        const pageCanvasHeight = Math.floor((canvas.width / contentWidth) * contentHeight);
        const totalPages = Math.ceil(canvas.height / pageCanvasHeight);

        for (let page = 0; page < totalPages; page++) {
            if (page > 0) {
                pdf.addPage();
            }

            const sourceY = page * pageCanvasHeight;
            const sourceHeight = Math.min(pageCanvasHeight, canvas.height - sourceY);
            const renderedHeight = (sourceHeight * contentWidth) / canvas.width;

            // Auxiliary canvas for clean, opaque white page slice
            const pageCanvas = document.createElement('canvas');
            pageCanvas.width = canvas.width;
            pageCanvas.height = sourceHeight;
            const pageCtx = pageCanvas.getContext('2d');

            if (pageCtx) {
                pageCtx.fillStyle = '#FFFFFF';
                pageCtx.fillRect(0, 0, pageCanvas.width, pageCanvas.height);
                pageCtx.drawImage(
                    canvas,
                    0,
                    sourceY,
                    canvas.width,
                    sourceHeight,
                    0,
                    0,
                    canvas.width,
                    sourceHeight
                );

                const imgData = pageCanvas.toDataURL('image/jpeg', 0.98);
                pdf.addImage(imgData, 'JPEG', margin, margin, contentWidth, renderedHeight);
            }
        }

        pdf.save(targetFilename);
    } catch (canvasErr) {
        console.warn('Canvas PDF export encountered an issue, executing direct PDF renderer fallback:', canvasErr);
        if (report) {
            exportToPdfDirect(report, targetFilename);
        } else {
            throw canvasErr;
        }
    } finally {
        if (container && container.parentNode) {
            container.parentNode.removeChild(container);
        }
    }
}

/**
 * Direct programmatic jsPDF renderer as robust backup or fallback
 */
export function exportToPdfDirect(report: ResearchReport, targetFilename: string): void {
    const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
    });

    const title = report.metadata?.title || report.title || 'Autonomous Scientific Literature Review';
    const question = report.metadata?.research_question || 'Literature Synthesis';
    const qualityScore = report.metadata?.quality_score !== undefined
        ? report.metadata.quality_score.toFixed(1)
        : (report.quality_score?.toFixed(1) ?? '85.0');
    const generatedDate = report.metadata?.generated_at
        ? new Date(report.metadata.generated_at).toLocaleDateString()
        : new Date().toLocaleDateString();

    let y = 20;
    const pageWidth = 210;
    const margin = 14;
    const maxLineWidth = pageWidth - 2 * margin;

    const checkPageBreak = (neededHeight: number) => {
        if (y + neededHeight > 275) {
            doc.addPage();
            y = 20;
        }
    };

    // Header Title
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(18);
    doc.setTextColor(15, 23, 42);
    const titleLines = doc.splitTextToSize(title, maxLineWidth);
    doc.text(titleLines, margin, y);
    y += titleLines.length * 7 + 4;

    // Metadata Subtitle
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(100, 116, 139);
    doc.text(`Generated: ${generatedDate} | Critic Quality Score: ${qualityScore}/100`, margin, y);
    y += 6;

    // Research Question
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(10);
    doc.setTextColor(51, 65, 85);
    const qLines = doc.splitTextToSize(`Target Question: "${question}"`, maxLineWidth);
    doc.text(qLines, margin, y);
    y += qLines.length * 5 + 6;

    // Divider
    doc.setDrawColor(226, 232, 240);
    doc.line(margin, y, pageWidth - margin, y);
    y += 8;

    // Executive Summary
    if (report.executive_summary) {
        checkPageBreak(30);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 42);
        doc.text('Executive Summary', margin, y);
        y += 6;

        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9.5);
        doc.setTextColor(30, 41, 59);
        const sumLines = doc.splitTextToSize(report.executive_summary, maxLineWidth);
        doc.text(sumLines, margin, y);
        y += sumLines.length * 4.5 + 8;
    }

    // Evidence Comparison Matrix
    const matrixRows = report.comparison_matrix || report.comparative_matrix || [];
    if (matrixRows.length > 0) {
        checkPageBreak(40);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 42);
        doc.text('Evidence Comparison Matrix', margin, y);
        y += 7;

        for (const row of matrixRows) {
            checkPageBreak(25);
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(9.5);
            doc.setTextColor(217, 119, 6);
            doc.text(`[${row.paper_id}] ${row.title}`, margin, y);
            y += 4.5;

            doc.setFont('helvetica', 'normal');
            doc.setFontSize(8.5);
            doc.setTextColor(100, 116, 139);
            doc.text(`${formatAuthors(row.authors)} ${row.year ? `(${row.year})` : ''}`, margin + 2, y);
            y += 4;

            doc.setTextColor(30, 41, 59);
            doc.text(`Methodology: ${row.methodology || row.methodology_type || '-'} | Dataset: ${row.benchmark_dataset || row.dataset || '-'}`, margin + 2, y);
            y += 4;

            if (row.primary_metric) {
                doc.text(`Primary Metric: ${row.primary_metric}`, margin + 2, y);
                y += 4;
            }
            if (row.primary_limitation) {
                doc.setTextColor(185, 28, 28);
                doc.text(`Limitation: ${row.primary_limitation}`, margin + 2, y);
                y += 4;
            }
            y += 3;
        }
        y += 4;
    }

    // Thematic Synthesis
    const sections = report.thematic_sections || report.sections || [];
    if (sections.length > 0) {
        checkPageBreak(30);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 42);
        doc.text('Thematic Synthesis', margin, y);
        y += 7;

        for (const sec of sections) {
            checkPageBreak(25);
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(11);
            doc.setTextColor(30, 41, 59);
            doc.text(sec.title, margin, y);
            y += 5.5;

            doc.setFont('helvetica', 'normal');
            doc.setFontSize(9.5);
            doc.setTextColor(51, 65, 85);
            const prose = sec.synthesis_prose || sec.content || '';
            const proseLines = doc.splitTextToSize(prose, maxLineWidth);
            doc.text(proseLines, margin, y);
            y += proseLines.length * 4.5 + 4;

            if (sec.key_takeaways && sec.key_takeaways.length > 0) {
                for (const t of sec.key_takeaways) {
                    checkPageBreak(10);
                    const tLines = doc.splitTextToSize(`• ${t}`, maxLineWidth - 4);
                    doc.text(tLines, margin + 4, y);
                    y += tLines.length * 4.5;
                }
                y += 4;
            }
        }
        y += 4;
    }

    // Conflicting Debates
    const debates = report.conflicting_findings_and_debates || report.conflicting_debates || report.debates || [];
    if (debates.length > 0) {
        checkPageBreak(30);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 42);
        doc.text('Scientific & Methodological Debates', margin, y);
        y += 7;

        for (const d of debates) {
            checkPageBreak(30);
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(10.5);
            doc.setTextColor(15, 23, 42);
            doc.text(`Debate: ${d.topic}`, margin, y);
            y += 5;

            doc.setFont('helvetica', 'normal');
            doc.setFontSize(9);
            doc.setTextColor(29, 78, 216);
            const pALines = doc.splitTextToSize(`Perspective A: ${d.perspective_a}`, maxLineWidth);
            doc.text(pALines, margin, y);
            y += pALines.length * 4 + 2;

            doc.setTextColor(185, 28, 28);
            const pBLines = doc.splitTextToSize(`Perspective B: ${d.perspective_b}`, maxLineWidth);
            doc.text(pBLines, margin, y);
            y += pBLines.length * 4 + 2;

            doc.setTextColor(4, 120, 87);
            const critLines = doc.splitTextToSize(`Evaluation: ${d.critical_evaluation}`, maxLineWidth);
            doc.text(critLines, margin, y);
            y += critLines.length * 4 + 5;
        }
        y += 4;
    }

    // Actionable Gaps
    const gaps = report.actionable_research_gaps || report.actionable_gaps || report.research_gaps || [];
    if (gaps.length > 0) {
        checkPageBreak(30);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 42);
        doc.text('Actionable Research Gaps & Future Directions', margin, y);
        y += 7;

        for (const g of gaps) {
            checkPageBreak(25);
            const prio = (g.importance || g.priority || 'HIGH').toUpperCase();
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(9.5);
            doc.setTextColor(prio === 'HIGH' ? 185 : 217, prio === 'HIGH' ? 28 : 119, prio === 'HIGH' ? 28 : 6);
            doc.text(`[${prio} PRIORITY] ${g.gap_id || 'GAP'}`, margin, y);
            y += 4.5;

            doc.setFont('helvetica', 'normal');
            doc.setFontSize(9);
            doc.setTextColor(15, 23, 42);
            const dLines = doc.splitTextToSize(g.description, maxLineWidth);
            doc.text(dLines, margin, y);
            y += dLines.length * 4 + 2;

            doc.setTextColor(21, 128, 61);
            const mLines = doc.splitTextToSize(`Recommended: ${g.recommended_methodology}`, maxLineWidth);
            doc.text(mLines, margin, y);
            y += mLines.length * 4 + 5;
        }
        y += 4;
    }

    // References & Bibliography
    const bib = report.bibliography || [];
    if (bib.length > 0) {
        checkPageBreak(30);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(13);
        doc.setTextColor(15, 23, 42);
        doc.text('References & Bibliography', margin, y);
        y += 7;

        for (const b of bib) {
            checkPageBreak(15);
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(8.5);
            doc.setTextColor(15, 23, 42);
            const auth = formatAuthors(b.authors);
            const yr = b.year ? `(${b.year})` : '';
            const venue = b.venue || b.source ? ` ${b.venue || b.source}.` : '';
            const doi = b.doi ? ` DOI: ${b.doi}` : '';
            const entryText = `[${b.paper_id}] ${auth} ${yr}. "${b.title}".${venue}${doi}`;
            const entryLines = doc.splitTextToSize(entryText, maxLineWidth);
            doc.text(entryLines, margin, y);
            y += entryLines.length * 4 + 3;
        }
    }

    doc.save(targetFilename);
}

/**
 * Unified Export Dispatcher
 */
export async function exportReport(
    format: 'md' | 'pdf' | 'docx' | 'bib',
    report: ResearchReport,
    elementId: string = 'synthesis-output',
    baseFilename?: string
): Promise<void> {
    const title = report.metadata?.title || report.title || baseFilename;
    const finalFilename = generateExportFilename(title, format);

    switch (format) {
        case 'md':
            exportToMarkdown(report, finalFilename);
            break;
        case 'pdf':
            await exportToPdf(report || elementId, finalFilename);
            break;
        case 'docx':
            await exportToDocx(report, finalFilename);
            break;
        case 'bib': {
            const items = (report.bibliography && report.bibliography.length > 0)
                ? report.bibliography
                : (report.comparison_matrix || []);
            exportToBibTeX(items, finalFilename);
            break;
        }
    }
}