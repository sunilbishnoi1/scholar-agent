import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getProjectById } from '../api/client';
import { Typography, Box, CircularProgress, Alert, Paper, Divider, Button, Menu, MenuItem } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import jsPDF from 'jspdf';
import { useState } from 'react';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import { Document, Packer, Paragraph, TextRun, HeadingLevel, BorderStyle } from 'docx';
import type { Root, Content } from 'mdast';
import { saveAs } from 'file-saver';


type HeadingValue = typeof HeadingLevel[keyof typeof HeadingLevel];

// Safely and recursively extracts text from any Markdown node.
const extractText = (node: Content): string => {
    if (node.type === 'text') {
        return node.value;
    }
    if ('children' in node && Array.isArray(node.children)) {
        return node.children.map(child => extractText(child as Content)).join('');
    }
    return '';
};

// Returns the correct HeadingLevel enum member from the docx library.
const getHeadingLevel = (depth: number): HeadingValue => {
    const levels: Record<number, HeadingValue> = {
        1: HeadingLevel.HEADING_1,
        2: HeadingLevel.HEADING_2,
        3: HeadingLevel.HEADING_3,
        4: HeadingLevel.HEADING_4,
        5: HeadingLevel.HEADING_5,
        6: HeadingLevel.HEADING_6,
    };
    return levels[depth] || HeadingLevel.HEADING_6;
};

// Traverses the Markdown tree and converts each node to a docx Paragraph.
const markdownToDocxElements = (markdown: string): Paragraph[] => {
    const tree = unified().use(remarkParse).parse(markdown) as Root;
    const elements: Paragraph[] = [];

    const processNode = (node: Content) => {
        switch (node.type) {
            case 'heading': {
                const text = extractText(node);
                const level = getHeadingLevel(node.depth);
                elements.push(new Paragraph({ text, heading: level }));
                break;
            }
            case 'paragraph': {
                const runs: TextRun[] = [];
                node.children.forEach(child => {
                    const text = extractText(child as Content);
                    if (child.type === 'strong') {
                        runs.push(new TextRun({ text, bold: true }));
                    } else if (child.type === 'emphasis') {
                        runs.push(new TextRun({ text, italics: true }));
                    } else {
                        runs.push(new TextRun(text));
                    }
                });
                elements.push(new Paragraph({ children: runs }));
                break;
            }
            case 'list': {
                node.children.forEach(listItem => {
                    const text = extractText(listItem);
                    elements.push(new Paragraph({
                        text: text,
                        bullet: { level: 0 },
                    }));
                });
                break;
            }
            case 'thematicBreak': {
                elements.push(new Paragraph({
                    border: { bottom: { color: "auto", space: 1, style: BorderStyle.SINGLE, size: 6 } },
                }));
                break;
            }
            default:
                break;
        }
    };

    tree.children.forEach(processNode);
    return elements;
};



const ProjectDetailsPage = () => {
    const { projectId } = useParams<{ projectId: string }>();
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const open = Boolean(anchorEl);

    const { data: project, isLoading, error } = useQuery({
        queryKey: ['project', projectId],
        queryFn: () => getProjectById(projectId!),
        enabled: !!projectId,
    });

    const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const exportToPdf = () => {
        const input = document.getElementById('synthesis-output');
        if (input) {
            const pdf = new jsPDF('p', 'mm', 'a4');
            
            // Use the html method for better control over the output
            pdf.html(input, {
                callback: function (doc) {
                    doc.save('literature-review.pdf');
                    handleClose(); // Close the menu after saving
                },
                // Set margins for the PDF
                margin: [15, 15, 15, 15], // [top, right, bottom, left]
                // Enable auto-paging to handle long content
                autoPaging: 'text',
                // Set a fixed width for the content to ensure consistent layout
                width: 180, // A4 width is 210mm, with 15mm margins on each side, content width is 180mm
                // Define a wide virtual window for html2canvas to render in, avoiding layout issues from small screens
                windowWidth: 1000,
            });
        }
    };

    const exportToDocx = () => {
        const synthesizerPlan = project?.agent_plans.find(p => p.agent_type === 'synthesizer');
        const synthesisOutput = synthesizerPlan?.plan_steps[0]?.output.response || "No synthesis available.";

        const docElements = markdownToDocxElements(synthesisOutput);

        const doc = new Document({
            sections: [
                {
                    children: docElements,
                },
            ],
        });

        Packer.toBlob(doc).then(blob => {
            saveAs(blob, "literature-review.docx");
        });

        handleClose();
    };


    if (isLoading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box sx={{ px: { xs: 2, sm: 3 }, py: 4 }}>
                <Alert severity="error" variant="filled">Failed to load project details: {error.message}</Alert>
            </Box>
        );
    }

    if (!project) {
        return (
            <Box sx={{ px: { xs: 2, sm: 3 }, py: 4 }}>
                <Alert severity="warning" variant="filled">Project not found.</Alert>
            </Box>
        );
    }

    const synthesizerPlan = project.agent_plans.find(p => p.agent_type === 'synthesizer');
    const synthesisOutput = synthesizerPlan?.plan_steps[0]?.output.response || "No synthesis available.";

    return (
        <Box sx={{ px: { xs: 1, sm: 2, md: 16 }, py: { xs: 2, sm: 4 } }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: { xs: 6, sm: 5 } }}>
                <Button
                    component={Link}
                    to="/"
                    startIcon={<ArrowBackIcon />}
                    sx={{
                        fontWeight: 600,
                        fontSize: { xs: '0.875rem', sm: '1rem' }
                    }}
                >
                    Back to Projects
                </Button>
                <Button
                    id="export-button"
                    aria-controls={open ? 'export-menu' : undefined}
                    aria-haspopup="true"
                    aria-expanded={open ? 'true' : undefined}
                    onClick={handleClick}
                    variant="contained"
                    sx={{marginTop:1}}
                >
                    Export
                </Button>
                <Menu
                    id="export-menu"
                    anchorEl={anchorEl}
                    open={open}
                    onClose={handleClose}
                    MenuListProps={{
                        'aria-labelledby': 'export-button',
                    }}
                >
                    <MenuItem onClick={exportToPdf}>Export as PDF</MenuItem>
                    <MenuItem onClick={exportToDocx}>Export as DOCX</MenuItem>
                </Menu>
            </Box>

            <Box className="bg-white/60 backdrop-blur-lg border border-gray-200/50 p-0 rounded-2xl" sx={{ mt: 2 }}>
                <Typography
                    variant="h4"
                    component="h1"
                    className="font-bold text-slate-800 mb-0"
                    sx={{
                        fontSize: { xs: '1.5rem', sm: '1.875rem', md: '2.125rem' },
                        px: { xs: 2, sm: 4 },
                        pt: { xs: 2, sm: 4 }
                    }}
                >
                    {project.title}
                </Typography>
                <Typography
                    variant="h6"
                    color="text.secondary"
                    className="italic mb-6"
                    sx={{
                        fontSize: { xs: '1rem', sm: '1.125rem', md: '1.25rem' },
                        p: { xs: 2, sm: 4 }
                    }}
                >
                    "{project.research_question}"
                </Typography>

                <Paper
                    elevation={0}
                    className="bg-slate-50/50 rounded-xl border border-slate-200/50"
                    sx={{
                        p: { xs: 2, sm: 4, md: 6 },
                        mb: { xs: 4, sm: 6, md: 8 },
                        mx: { xs: 1, sm: 2, md: 0 }
                    }}
                >
                    <Typography
                        variant="h5"
                        className="!font-bold text-slate-800 !mb-4"
                        sx={{ fontSize: { xs: '1.125rem', sm: '1.25rem', md: '1.5rem' } }}
                    >
                        Synthesized Literature Review
                    </Typography>
                    <Divider className="!mb-4" />
                    <Box
                        id="synthesis-output"
                        className="prose prose-slate max-w-none prose-headings:font-semibold prose-a:text-blue-600 hover:prose-a:text-blue-800"
                        sx={{
                            '& h1': { fontSize: { xs: '1.25rem', sm: '1.5rem' } },
                            '& h2': { fontSize: { xs: '1.125rem', sm: '1.375rem' } },
                            '& h3': { fontSize: { xs: '1rem', sm: '1.25rem' } },
                            '& p': { fontSize: { xs: '0.875rem', sm: '1rem' } },
                            '& li': { fontSize: { xs: '0.875rem', sm: '1rem' } }
                        }}
                    >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {synthesisOutput}
                        </ReactMarkdown>
                    </Box>
                </Paper>

                <Box sx={{ px: { xs: 2, sm: 4, md: 2 } }}>
                    <Typography
                        variant="h5"
                        className="!font-bold text-slate-800 !mb-4"
                        sx={{ fontSize: { xs: '1.125rem', sm: '1.25rem', md: '1.5rem' } }}
                    >
                        Analyzed Papers ({project.paper_references.length})
                    </Typography>
                    <div className="space-y-4">
                        {project.paper_references
                            .sort((a, b) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0))
                            .map((paper) => (
                                <Paper
                                    key={paper.id}
                                    variant="outlined"
                                    className="transition-shadow duration-300 hover:shadow-md rounded-lg bg-white/50"
                                    sx={{ p: { xs: 2, sm: 3, md: 4 } }}
                                >
                                    <Box className="flex justify-between items-start">
                                        <Typography
                                            variant="h6"
                                            className="!font-semibold flex-1 pr-4"
                                            sx={{ fontSize: { xs: '0.95rem', sm: '1.125rem', md: '1.25rem' } }}
                                        >
                                            {paper.title}
                                        </Typography>
                                        <Typography
                                            className="font-medium text-indigo-600 bg-indigo-100/80 px-2 py-1 rounded-md"
                                            sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
                                        >
                                            {paper.relevance_score.toFixed(1)} / 100
                                        </Typography>
                                    </Box>
                                    <Typography
                                        variant="body2"
                                        color="text.secondary"
                                        className="mb-2 mt-1"
                                        sx={{ fontSize: { xs: '0.75rem', sm: '0.875rem' } }}
                                    >
                                        {paper.authors.join(', ')}
                                    </Typography>
                                    <a
                                        href={paper.url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="text-blue-500 hover:underline mt-2 block font-semibold"
                                        style={{ fontSize: '0.875rem' }}
                                    >
                                        Read Paper
                                    </a>
                                </Paper>
                            ))}
                    </div>
                </Box>
            </Box>
        </Box>
    );
};

export default ProjectDetailsPage;