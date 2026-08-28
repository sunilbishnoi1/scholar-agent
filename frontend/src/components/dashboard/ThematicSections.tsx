import React, { useState, useMemo } from 'react';
import {
    Box,
    Typography,
    Paper,
    Chip,
    Popover,
    Button,
    Divider,
    IconButton,
} from '@mui/material';
import { styled } from '@mui/system';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import AutoStoriesIcon from '@mui/icons-material/AutoStories';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import CloseIcon from '@mui/icons-material/Close';
import type { ThematicSection, BibliographyItem, PaperReference } from '../../types';
import { formatAuthors } from '../../utils/exportEngine';

const SectionContainer = styled(Paper)({
    backgroundColor: 'rgba(24, 24, 27, 0.7)',
    backdropFilter: 'blur(20px) saturate(180%)',
    WebkitBackdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '16px',
    padding: '28px',
    marginBottom: '24px',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
    color: '#F4F4F5',
    transition: 'border-color 0.3s ease',
    '&:hover': {
        borderColor: 'rgba(255, 185, 0, 0.25)',
    },
});

const TakeawaysCard = styled(Box)({
    backgroundColor: 'rgba(255, 185, 0, 0.04)',
    border: '1px solid rgba(255, 185, 0, 0.2)',
    borderRadius: '12px',
    padding: '18px 20px',
    marginTop: '24px',
});

const CitationChip = styled('span')({
    display: 'inline-flex',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 245, 200, 0.12)',
    color: '#00F5C8',
    border: '1px solid rgba(0, 245, 200, 0.3)',
    borderRadius: '6px',
    padding: '1px 6px',
    fontSize: '0.8rem',
    fontWeight: 700,
    fontFamily: 'var(--font-mono, monospace)',
    cursor: 'pointer',
    margin: '0 2px',
    transition: 'all 0.2s ease',
    '&:hover': {
        backgroundColor: 'rgba(0, 245, 200, 0.25)',
        borderColor: '#00F5C8',
        boxShadow: '0 0 10px rgba(0, 245, 200, 0.3)',
        transform: 'translateY(-1px)',
    },
});

interface ThematicSectionsProps {
    sections: ThematicSection[];
    bibliography?: BibliographyItem[];
    paperReferences?: PaperReference[];
    onPaperClick?: (paperId: string) => void;
}

interface SelectedCitation {
    anchor: string;
    paperId: string;
    sectionTag?: string;
    paper?: BibliographyItem | PaperReference;
}

export const ThematicSections: React.FC<ThematicSectionsProps> = ({
    sections = [],
    bibliography = [],
    paperReferences = [],
}) => {
    const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
    const [selectedCitation, setSelectedCitation] = useState<SelectedCitation | null>(null);
    const [copied, setCopied] = useState(false);

    // Build lookup dictionary for bibliography papers by paper_id or prefix
    const paperLookup = useMemo(() => {
        const map = new Map<string, BibliographyItem | PaperReference>();
        for (const item of bibliography) {
            map.set(item.paper_id.toLowerCase(), item);
            const cleanId = item.paper_id.replace(/^ref_/, '').toLowerCase();
            map.set(cleanId, item);
        }
        for (const ref of paperReferences) {
            const pid = (ref.paper_id || ref.id).toLowerCase();
            if (!map.has(pid)) {
                map.set(pid, ref);
            }
            const cleanRef = pid.replace(/^ref_/, '').toLowerCase();
            if (!map.has(cleanRef)) {
                map.set(cleanRef, ref);
            }
        }
        return map;
    }, [bibliography, paperReferences]);

    const handleCitationClick = (event: React.MouseEvent<HTMLElement>, anchorToken: string) => {
        event.stopPropagation();
        const cleanToken = anchorToken.replace(/^\[|\]$/g, '');
        const [rawPaperId, sectionTag] = cleanToken.split('#');
        const paperId = rawPaperId.trim();

        const matchedPaper = paperLookup.get(paperId.toLowerCase()) ||
            paperLookup.get(paperId.replace(/^ref_/, '').toLowerCase());

        setSelectedCitation({
            anchor: cleanToken,
            paperId,
            sectionTag,
            paper: matchedPaper,
        });
        setAnchorEl(event.currentTarget);
        setCopied(false);
    };

    const handlePopoverClose = () => {
        setAnchorEl(null);
        setSelectedCitation(null);
    };

    const handleCopyCitation = () => {
        if (!selectedCitation) return;
        const text = selectedCitation.paper
            ? `[${selectedCitation.paperId}] ${formatAuthors(selectedCitation.paper.authors)}. "${selectedCitation.paper.title}"`
            : `[${selectedCitation.anchor}]`;
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    /**
     * Custom text renderer replacing [ref_X#secY] and [ref_X] anchors with clickable chips
     */
    const renderProseWithCitations = (prose: string) => {
        // Regex matches [ref_X#secY] or [ref_X] or [ref_1]
        const citationRegex = /(\[ref_[a-zA-Z0-9_\-#.]+\]|\[ref_[0-9]+#[a-zA-Z0-9_]+\])/g;
        const parts = prose.split(citationRegex);

        return parts.map((part, index) => {
            if (citationRegex.test(part)) {
                return (
                    <CitationChip
                        key={index}
                        onClick={(e) => handleCitationClick(e, part)}
                        title={`View citation context for ${part}`}
                    >
                        {part}
                    </CitationChip>
                );
            }
            return <span key={index}>{part}</span>;
        });
    };

    if (!sections || sections.length === 0) {
        return (
            <SectionContainer elevation={0}>
                <Box sx={{ textAlign: 'center', py: 4, color: '#71717A' }}>
                    <AutoStoriesIcon sx={{ fontSize: '3rem', opacity: 0.3, mb: 1 }} />
                    <Typography variant="h6" sx={{ color: '#A1A1AA' }}>
                        No thematic sections synthesized yet
                    </Typography>
                    <Typography variant="body2">
                        The Synthesizer agent will draft thematic narrative reviews here.
                    </Typography>
                </Box>
            </SectionContainer>
        );
    }

    return (
        <Box>
            {sections.map((section, idx) => (
                <SectionContainer key={section.theme_id || idx} elevation={0}>
                    {/* Theme Header */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
                        <AutoStoriesIcon sx={{ color: '#FFB900', fontSize: '1.5rem' }} />
                        <Typography
                            variant="h5"
                            sx={{
                                fontWeight: 800,
                                color: '#F4F4F5',
                                letterSpacing: '-0.02em',
                                fontSize: { xs: '1.25rem', md: '1.5rem' },
                            }}
                        >
                            {section.title}
                        </Typography>
                    </Box>

                    {/* Synthesis Narrative Prose with ReactMarkdown & RemarkGFM */}
                    <Box
                        sx={{
                            color: '#D4D4D8',
                            lineHeight: 1.85,
                            fontSize: '1rem',
                            fontFamily: 'var(--font-content, sans-serif)',
                            '& p': { mb: 2 },
                            '& h4, & h5': { color: '#F4F4F5', mt: 2, mb: 1, fontWeight: 700 },
                            '& ul, & ol': { pl: 3, mb: 2 },
                            '& li': { mb: 0.5 },
                            '& blockquote': {
                                borderLeft: '3px solid #FFB900',
                                pl: 2,
                                py: 0.5,
                                my: 2,
                                backgroundColor: 'rgba(255, 185, 0, 0.05)',
                                fontStyle: 'italic',
                                color: '#E4E4E7',
                            },
                        }}
                    >
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                p: ({ children }) => {
                                    // Process text children for citation token replacements
                                    if (typeof children === 'string') {
                                        return <p>{renderProseWithCitations(children)}</p>;
                                    }
                                    return <p>{children}</p>;
                                },
                            }}
                        >
                            {section.synthesis_prose || section.content || ''}
                        </ReactMarkdown>
                    </Box>

                    {/* Key Takeaways Card */}
                    {section.key_takeaways && section.key_takeaways.length > 0 && (
                        <TakeawaysCard>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                <CheckCircleIcon sx={{ color: '#00F5C8', fontSize: '1.2rem' }} />
                                <Typography
                                    variant="subtitle2"
                                    sx={{
                                        color: '#FFB900',
                                        fontWeight: 800,
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.05em',
                                        fontSize: '0.8rem',
                                    }}
                                >
                                    Key Findings & Actionable Takeaways
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                {section.key_takeaways.map((takeaway, tIdx) => (
                                    <Box key={tIdx} sx={{ display: 'flex', alignItems: 'flex-start', gap: 1.5 }}>
                                        <Box
                                            component="span"
                                            sx={{
                                                width: 6,
                                                height: 6,
                                                borderRadius: '50%',
                                                backgroundColor: '#00F5C8',
                                                mt: 1,
                                                flexShrink: 0,
                                            }}
                                        />
                                        <Typography variant="body2" sx={{ color: '#F4F4F5', lineHeight: 1.6 }}>
                                            {takeaway}
                                        </Typography>
                                    </Box>
                                ))}
                            </Box>
                        </TakeawaysCard>
                    )}
                </SectionContainer>
            ))}

            {/* Interactive Citation Popover */}
            <Popover
                open={Boolean(anchorEl)}
                anchorEl={anchorEl}
                onClose={handlePopoverClose}
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'left',
                }}
                transformOrigin={{
                    vertical: 'top',
                    horizontal: 'left',
                }}
                PaperProps={{
                    sx: {
                        backgroundColor: '#18181B',
                        border: '1px solid rgba(0, 245, 200, 0.3)',
                        borderRadius: '12px',
                        boxShadow: '0 12px 32px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 245, 200, 0.1)',
                        maxWidth: '420px',
                        p: 2.5,
                        color: '#F4F4F5',
                    },
                }}
            >
                {selectedCitation && (
                    <Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                            <Chip
                                label={`Citation [${selectedCitation.anchor}]`}
                                size="small"
                                sx={{
                                    backgroundColor: 'rgba(0, 245, 200, 0.15)',
                                    color: '#00F5C8',
                                    fontWeight: 800,
                                    fontSize: '0.75rem',
                                    border: '1px solid rgba(0, 245, 200, 0.4)',
                                }}
                            />
                            <IconButton size="small" onClick={handlePopoverClose} sx={{ color: '#71717A', p: 0.5 }}>
                                <CloseIcon sx={{ fontSize: '1rem' }} />
                            </IconButton>
                        </Box>

                        {selectedCitation.paper ? (
                            <Box sx={{ mt: 1.5 }}>
                                <Typography variant="subtitle2" sx={{ fontWeight: 800, color: '#F4F4F5', lineHeight: 1.4, mb: 0.5 }}>
                                    {selectedCitation.paper.title}
                                </Typography>
                                <Typography variant="caption" sx={{ color: '#A1A1AA', display: 'block', mb: 1 }}>
                                    {formatAuthors(selectedCitation.paper.authors)} {selectedCitation.paper.year ? `(${selectedCitation.paper.year})` : ''}
                                </Typography>

                                {selectedCitation.sectionTag && (
                                    <Box sx={{ backgroundColor: 'rgba(255, 185, 0, 0.08)', p: 1, borderRadius: '6px', border: '1px solid rgba(255, 185, 0, 0.2)', mb: 1.5 }}>
                                        <Typography variant="caption" sx={{ color: '#FFB900', fontWeight: 700 }}>
                                            Referenced Section Anchor:
                                        </Typography>
                                        <Typography variant="caption" sx={{ color: '#E4E4E7', display: 'block', fontFamily: 'var(--font-mono)' }}>
                                            § {selectedCitation.sectionTag.replace(/_/g, ' ')}
                                        </Typography>
                                    </Box>
                                )}

                                <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.08)', my: 1.5 }} />

                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Button
                                        size="small"
                                        startIcon={<ContentCopyIcon sx={{ fontSize: '0.9rem !important' }} />}
                                        onClick={handleCopyCitation}
                                        sx={{
                                            color: copied ? '#00B894' : '#A1A1AA',
                                            textTransform: 'none',
                                            fontSize: '0.75rem',
                                            fontWeight: 600,
                                        }}
                                    >
                                        {copied ? 'Copied Reference!' : 'Copy Reference'}
                                    </Button>

                                    {((selectedCitation.paper as BibliographyItem).pdf_url || (selectedCitation.paper as PaperReference).url || (selectedCitation.paper as BibliographyItem).doi) && (
                                        <Button
                                            size="small"
                                            component="a"
                                            href={
                                                (selectedCitation.paper as BibliographyItem).pdf_url ||
                                                (selectedCitation.paper as PaperReference).url ||
                                                ((selectedCitation.paper as BibliographyItem).doi ? `https://doi.org/${(selectedCitation.paper as BibliographyItem).doi}` : '#')
                                            }
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            endIcon={<OpenInNewIcon sx={{ fontSize: '0.9rem !important' }} />}
                                            sx={{
                                                backgroundColor: '#FFB900',
                                                color: '#09090B',
                                                fontWeight: 700,
                                                fontSize: '0.75rem',
                                                textTransform: 'none',
                                                borderRadius: '6px',
                                                '&:hover': { backgroundColor: '#E6A600' },
                                            }}
                                        >
                                            View Source
                                        </Button>
                                    )}
                                </Box>
                            </Box>
                        ) : (
                            <Box sx={{ mt: 1.5 }}>
                                <Typography variant="body2" sx={{ color: '#D4D4D8' }}>
                                    Grounding evidence anchored to <strong>{selectedCitation.paperId}</strong> in the study corpus.
                                </Typography>
                                {selectedCitation.sectionTag && (
                                    <Typography variant="caption" sx={{ color: '#FFB900', display: 'block', mt: 0.5 }}>
                                        Section chunk: {selectedCitation.sectionTag}
                                    </Typography>
                                )}
                            </Box>
                        )}
                    </Box>
                )}
            </Popover>
        </Box>
    );
};

export default ThematicSections;