import React, { useState, useMemo } from 'react';
import {
    Box,
    Typography,
    Paper,
    Chip,
    Button,
    TextField,
    InputAdornment,
} from '@mui/material';
import { styled } from '@mui/system';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import SearchIcon from '@mui/icons-material/Search';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import type { ResearchGapItem, BibliographyItem } from '../../types';

const GapCard = styled(Paper)<{ prioritylevel: string }>(({ prioritylevel }) => {
    const borderColors: Record<string, string> = {
        high: 'rgba(239, 68, 68, 0.3)',
        medium: 'rgba(245, 158, 11, 0.3)',
        low: 'rgba(59, 130, 246, 0.3)',
    };
    const glowColors: Record<string, string> = {
        high: 'rgba(239, 68, 68, 0.08)',
        medium: 'rgba(245, 158, 11, 0.08)',
        low: 'rgba(59, 130, 246, 0.08)',
    };

    const border = borderColors[prioritylevel] || borderColors.high;
    const glow = glowColors[prioritylevel] || glowColors.high;

    return {
        backgroundColor: 'rgba(24, 24, 27, 0.75)',
        backdropFilter: 'blur(20px) saturate(180%)',
        WebkitBackdropFilter: 'blur(20px) saturate(180%)',
        border: `1px solid ${border}`,
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '20px',
        boxShadow: `0 8px 32px 0 rgba(0, 0, 0, 0.3), 0 0 20px ${glow}`,
        color: '#F4F4F5',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: `0 12px 36px 0 rgba(0, 0, 0, 0.4), 0 0 28px ${glow}`,
        },
    };
});

const MethodologyBox = styled(Box)({
    backgroundColor: 'rgba(0, 245, 200, 0.04)',
    border: '1px solid rgba(0, 245, 200, 0.2)',
    borderRadius: '12px',
    padding: '16px 20px',
    marginTop: '16px',
    marginBottom: '16px',
});

const SearchInput = styled(TextField)({
    '& .MuiOutlinedInput-root': {
        backgroundColor: '#18181B',
        borderRadius: '10px',
        color: '#F4F4F5',
        fontSize: '0.875rem',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        '& fieldset': { border: 'none' },
        '&:hover': {
            backgroundColor: '#27272F',
        },
        '&.Mui-focused': {
            backgroundColor: '#18181B',
            border: '1px solid #FFB900',
        },
    },
    '& .MuiInputBase-input::placeholder': {
        color: '#71717A',
        opacity: 1,
    },
});

interface ResearchGapViewerProps {
    gaps: ResearchGapItem[];
    bibliography?: BibliographyItem[];
}

export const ResearchGapViewer: React.FC<ResearchGapViewerProps> = ({ gaps = [] }) => {
    const [selectedPriority, setSelectedPriority] = useState<string>('all');
    const [searchTerm, setSearchTerm] = useState('');
    const [copiedGapId, setCopiedGapId] = useState<string | null>(null);

    const counts = useMemo(() => {
        return {
            all: gaps.length,
            high: gaps.filter((g) => (g.importance || g.priority || 'high').toLowerCase() === 'high').length,
            medium: gaps.filter((g) => (g.importance || g.priority || 'high').toLowerCase() === 'medium').length,
            low: gaps.filter((g) => (g.importance || g.priority || 'high').toLowerCase() === 'low').length,
        };
    }, [gaps]);

    const filteredGaps = useMemo(() => {
        return gaps.filter((gap) => {
            const prio = (gap.importance || gap.priority || 'high').toLowerCase();
            if (selectedPriority !== 'all' && prio !== selectedPriority) {
                return false;
            }
            if (!searchTerm.trim()) return true;
            const term = searchTerm.toLowerCase();
            return (
                (gap.title || '').toLowerCase().includes(term) ||
                gap.description.toLowerCase().includes(term) ||
                gap.recommended_methodology.toLowerCase().includes(term) ||
                (gap.gap_id || '').toLowerCase().includes(term) ||
                (gap.grounding_paper_ids || gap.grounding_papers || []).some((id: unknown) =>
                    (typeof id === 'object' && id !== null ? (id as { paper_id?: string; title?: string }).paper_id || (id as { paper_id?: string; title?: string }).title || '' : String(id)).toLowerCase().includes(term)
                )
            );
        });
    }, [gaps, selectedPriority, searchTerm]);

    const handleCopyGap = (gap: ResearchGapItem) => {
        const text = `Research Gap [${(gap.importance || 'HIGH').toUpperCase()}]: ${gap.title || gap.description}\nRecommended Methodology: ${gap.recommended_methodology}\nGrounding Papers: ${(gap.grounding_paper_ids || []).join(', ')}`;
        navigator.clipboard.writeText(text);
        setCopiedGapId(gap.gap_id || gap.id || 'copied');
        setTimeout(() => setCopiedGapId(null), 2000);
    };

    if (!gaps || gaps.length === 0) {
        return (
            <Paper
                elevation={0}
                sx={{
                    backgroundColor: 'rgba(24, 24, 27, 0.7)',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    borderRadius: '16px',
                    p: 6,
                    textAlign: 'center',
                    color: '#71717A',
                }}
            >
                <TrendingUpIcon sx={{ fontSize: '3.5rem', opacity: 0.3, mb: 1 }} />
                <Typography variant="h6" sx={{ color: '#A1A1AA', fontWeight: 600 }}>
                    No research gaps synthesized yet
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                    Open research directions will be extracted and grounded in paper limitations during synthesis.
                </Typography>
            </Paper>
        );
    }

    return (
        <Box>
            {/* Header & Filter Controls */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, flexWrap: 'wrap', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <TrendingUpIcon sx={{ color: '#FFB900', fontSize: '1.75rem' }} />
                    <Box>
                        <Typography variant="h5" sx={{ fontWeight: 800, color: '#F4F4F5', letterSpacing: '-0.02em' }}>
                            Actionable Research Gaps & Future Directions
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#71717A' }}>
                            Grounded open problems with recommended methodological frameworks
                        </Typography>
                    </Box>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
                    {/* Priority Filter Tabs */}
                    <Chip
                        label={`All (${counts.all})`}
                        onClick={() => setSelectedPriority('all')}
                        sx={{
                            backgroundColor: selectedPriority === 'all' ? 'rgba(255, 185, 0, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                            color: selectedPriority === 'all' ? '#FFB900' : '#A1A1AA',
                            border: `1px solid ${selectedPriority === 'all' ? 'rgba(255, 185, 0, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    />
                    <Chip
                        label={`High Priority (${counts.high})`}
                        onClick={() => setSelectedPriority('high')}
                        sx={{
                            backgroundColor: selectedPriority === 'high' ? 'rgba(239, 68, 68, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                            color: selectedPriority === 'high' ? '#EF4444' : '#A1A1AA',
                            border: `1px solid ${selectedPriority === 'high' ? 'rgba(239, 68, 68, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    />
                    <Chip
                        label={`Medium (${counts.medium})`}
                        onClick={() => setSelectedPriority('medium')}
                        sx={{
                            backgroundColor: selectedPriority === 'medium' ? 'rgba(245, 158, 11, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                            color: selectedPriority === 'medium' ? '#F59E0B' : '#A1A1AA',
                            border: `1px solid ${selectedPriority === 'medium' ? 'rgba(245, 158, 11, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    />

                    <SearchInput
                        size="small"
                        placeholder="Search research gaps..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <SearchIcon sx={{ color: '#71717A', fontSize: '1.1rem' }} />
                                </InputAdornment>
                            ),
                        }}
                        sx={{ width: { xs: '100%', sm: '220px' } }}
                    />
                </Box>
            </Box>

            {/* Render Gap Cards */}
            {filteredGaps.map((gap, index) => {
                const priority = (gap.importance || gap.priority || 'high').toLowerCase();
                const isHigh = priority === 'high';
                const isMedium = priority === 'medium';
                const priorityColor = isHigh ? '#EF4444' : isMedium ? '#F59E0B' : '#3B82F6';
                const isCopied = copiedGapId === (gap.gap_id || gap.id || 'gap');

                const groundingList = (gap.grounding_paper_ids || gap.grounding_papers || []).map((id: unknown) =>
                    typeof id === 'object' && id !== null ? (id as { paper_id?: string; id?: string }).paper_id || (id as { paper_id?: string; id?: string }).id || 'Ref' : String(id)
                );

                return (
                    <GapCard key={gap.gap_id || gap.id || index} prioritylevel={priority} elevation={0}>
                        {/* Top row: Priority badge + Gap ID + Copy button */}
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <Chip
                                    label={`${priority.toUpperCase()} PRIORITY`}
                                    size="small"
                                    sx={{
                                        backgroundColor: `${priorityColor}20`,
                                        color: priorityColor,
                                        fontWeight: 800,
                                        fontSize: '0.75rem',
                                        border: `1px solid ${priorityColor}50`,
                                    }}
                                />
                                <Typography variant="caption" sx={{ color: '#71717A', fontFamily: 'var(--font-mono)', fontWeight: 700 }}>
                                    {gap.gap_id || `gap_${index + 1}`}
                                </Typography>
                            </Box>

                            <Button
                                size="small"
                                startIcon={<ContentCopyIcon sx={{ fontSize: '0.9rem !important' }} />}
                                onClick={() => handleCopyGap(gap)}
                                sx={{
                                    color: isCopied ? '#00B894' : '#A1A1AA',
                                    textTransform: 'none',
                                    fontSize: '0.75rem',
                                    fontWeight: 600,
                                }}
                            >
                                {isCopied ? 'Copied!' : 'Copy Summary'}
                            </Button>
                        </Box>

                        {/* Substantive Description (only render distinct domain title if not duplicate of description) */}
                        {(() => {
                            const hasDistinctTitle = Boolean(
                                gap.title &&
                                gap.title.trim() &&
                                gap.title.trim() !== gap.description.trim() &&
                                !gap.description.toLowerCase().startsWith(gap.title.toLowerCase().replace(/\.{3,}$/, '').trim()) &&
                                gap.title.length < gap.description.length * 0.75
                            );
                            return (
                                <>
                                    {hasDistinctTitle && (
                                        <Typography variant="h6" sx={{ fontWeight: 800, color: '#F4F4F5', mb: 1, lineHeight: 1.35 }}>
                                            {gap.title}
                                        </Typography>
                                    )}
                                    <Typography variant="body1" sx={{ color: '#E4E4E7', lineHeight: 1.7, fontSize: '0.975rem', mb: 1 }}>
                                        {gap.description}
                                    </Typography>
                                </>
                            );
                        })()}

                        {/* Actionable Recommended Methodology */}
                        <MethodologyBox>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                <LightbulbIcon sx={{ color: '#00F5C8', fontSize: '1.25rem' }} />
                                <Typography
                                    variant="subtitle2"
                                    sx={{
                                        color: '#00F5C8',
                                        fontWeight: 800,
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.05em',
                                        fontSize: '0.8rem',
                                    }}
                                >
                                    Actionable Methodology & Experimental Roadmap
                                </Typography>
                            </Box>
                            <Typography variant="body2" sx={{ color: '#F4F4F5', lineHeight: 1.7 }}>
                                {gap.recommended_methodology}
                            </Typography>
                        </MethodologyBox>

                        {/* Grounding Literature Chips */}
                        {groundingList.length > 0 && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mt: 1 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <MenuBookIcon sx={{ color: '#71717A', fontSize: '0.95rem' }} />
                                    <Typography variant="caption" sx={{ color: '#71717A', fontWeight: 700 }}>
                                        Substantiating Papers:
                                    </Typography>
                                </Box>
                                {groundingList.map((pid, pIdx) => (
                                    <Chip
                                        key={pIdx}
                                        label={`[${pid}]`}
                                        size="small"
                                        sx={{
                                            backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                            color: '#FFB900',
                                            fontWeight: 700,
                                            fontSize: '0.75rem',
                                            fontFamily: 'var(--font-mono)',
                                            border: '1px solid rgba(255, 185, 0, 0.2)',
                                        }}
                                    />
                                ))}
                            </Box>
                        )}
                    </GapCard>
                );
            })}
        </Box>
    );
};

export default ResearchGapViewer;