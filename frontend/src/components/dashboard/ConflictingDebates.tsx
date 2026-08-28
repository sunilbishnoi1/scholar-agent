import React, { useState, useMemo } from 'react';
import {
    Box,
    Typography,
    Paper,
    TextField,
    InputAdornment,
    Chip,
} from '@mui/material';
import { styled } from '@mui/system';
import SearchIcon from '@mui/icons-material/Search';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import BalanceIcon from '@mui/icons-material/Balance';
import type { ConflictingDebate } from '../../types';

const DebateContainer = styled(Paper)({
    backgroundColor: 'rgba(24, 24, 27, 0.7)',
    backdropFilter: 'blur(20px) saturate(180%)',
    WebkitBackdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '16px',
    padding: '28px',
    marginBottom: '24px',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
    color: '#F4F4F5',
});

const PerspectiveCard = styled(Box, {
    shouldForwardProp: (prop) => prop !== 'variantType',
})<{ variantType: 'a' | 'b' }>(({ variantType }) => ({
    backgroundColor: variantType === 'a' ? 'rgba(56, 189, 248, 0.04)' : 'rgba(244, 63, 94, 0.04)',
    border: `1px solid ${variantType === 'a' ? 'rgba(56, 189, 248, 0.25)' : 'rgba(244, 63, 94, 0.25)'}`,
    borderRadius: '14px',
    padding: '20px',
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    transition: 'all 0.3s ease',
    '&:hover': {
        transform: 'translateY(-2px)',
        borderColor: variantType === 'a' ? 'rgba(56, 189, 248, 0.5)' : 'rgba(244, 63, 94, 0.5)',
        boxShadow: `0 8px 24px ${variantType === 'a' ? 'rgba(56, 189, 248, 0.1)' : 'rgba(244, 63, 94, 0.1)'}`,
    },
}));

const EvaluationCard = styled(Box)({
    backgroundColor: 'rgba(16, 185, 129, 0.05)',
    border: '1px solid rgba(16, 185, 129, 0.25)',
    borderRadius: '14px',
    padding: '20px 24px',
    marginTop: '20px',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        width: '4px',
        height: '100%',
        backgroundColor: '#10B981',
    },
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

interface ConflictingDebatesProps {
    debates: ConflictingDebate[];
}

export const ConflictingDebates: React.FC<ConflictingDebatesProps> = ({ debates = [] }) => {
    const [searchTerm, setSearchTerm] = useState('');

    const filteredDebates = useMemo(() => {
        if (!searchTerm.trim()) return debates;
        const term = searchTerm.toLowerCase();
        return debates.filter(
            (d) =>
                d.topic.toLowerCase().includes(term) ||
                d.perspective_a.toLowerCase().includes(term) ||
                d.perspective_b.toLowerCase().includes(term) ||
                d.critical_evaluation.toLowerCase().includes(term)
        );
    }, [debates, searchTerm]);

    if (!debates || debates.length === 0) {
        return (
            <DebateContainer elevation={0}>
                <Box sx={{ textAlign: 'center', py: 6, color: '#71717A' }}>
                    <CompareArrowsIcon sx={{ fontSize: '3.5rem', opacity: 0.3, mb: 1 }} />
                    <Typography variant="h6" sx={{ color: '#A1A1AA', fontWeight: 600 }}>
                        No scientific controversies or debates detected
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                        The literature corpus currently exhibits strong methodological alignment with no polarized debates.
                    </Typography>
                </Box>
            </DebateContainer>
        );
    }

    return (
        <Box>
            {/* Header with Search */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, flexWrap: 'wrap', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <CompareArrowsIcon sx={{ color: '#FFB900', fontSize: '1.75rem' }} />
                    <Box>
                        <Typography variant="h5" sx={{ fontWeight: 800, color: '#F4F4F5', letterSpacing: '-0.02em' }}>
                            Methodological & Empirical Debates
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#71717A' }}>
                            Contrasting paradigms, competing empirical results, and architectural trade-offs
                        </Typography>
                    </Box>
                </Box>

                <SearchInput
                    size="small"
                    placeholder="Search debates..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    InputProps={{
                        startAdornment: (
                            <InputAdornment position="start">
                                <SearchIcon sx={{ color: '#71717A', fontSize: '1.1rem' }} />
                            </InputAdornment>
                        ),
                    }}
                    sx={{ width: { xs: '100%', sm: '260px' } }}
                />
            </Box>

            {/* List of Debates */}
            {filteredDebates.map((debate, index) => (
                <DebateContainer key={index} elevation={0}>
                    {/* Debate Topic Badge & Title */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                        <Chip
                            label={`Debate #${index + 1}`}
                            size="small"
                            sx={{
                                backgroundColor: 'rgba(255, 185, 0, 0.15)',
                                color: '#FFB900',
                                fontWeight: 800,
                                border: '1px solid rgba(255, 185, 0, 0.3)',
                            }}
                        />
                        <Typography variant="h6" sx={{ fontWeight: 800, color: '#F4F4F5', letterSpacing: '-0.01em' }}>
                            {debate.topic}
                        </Typography>
                    </Box>

                    {/* Side-by-Side Perspectives */}
                    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
                        {/* Perspective A */}
                        <PerspectiveCard variantType="a">
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                <Chip
                                    label="Perspective A"
                                    size="small"
                                    sx={{
                                        backgroundColor: 'rgba(56, 189, 248, 0.2)',
                                        color: '#38BDF8',
                                        fontWeight: 800,
                                        fontSize: '0.75rem',
                                    }}
                                />
                                <Typography variant="caption" sx={{ color: '#71717A', fontWeight: 600 }}>
                                    Thesis / Hypothesis 1
                                </Typography>
                            </Box>
                            <Typography variant="body2" sx={{ color: '#E4E4E7', lineHeight: 1.7, flex: 1 }}>
                                {debate.perspective_a}
                            </Typography>
                        </PerspectiveCard>

                        {/* Perspective B */}
                        <PerspectiveCard variantType="b">
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                <Chip
                                    label="Perspective B"
                                    size="small"
                                    sx={{
                                        backgroundColor: 'rgba(244, 63, 94, 0.2)',
                                        color: '#F43F5E',
                                        fontWeight: 800,
                                        fontSize: '0.75rem',
                                    }}
                                />
                                <Typography variant="caption" sx={{ color: '#71717A', fontWeight: 600 }}>
                                    Antithesis / Opposing View
                                </Typography>
                            </Box>
                            <Typography variant="body2" sx={{ color: '#E4E4E7', lineHeight: 1.7, flex: 1 }}>
                                {debate.perspective_b}
                            </Typography>
                        </PerspectiveCard>
                    </Box>

                    {/* Critical Evaluation Box */}
                    <EvaluationCard>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <BalanceIcon sx={{ color: '#10B981', fontSize: '1.25rem' }} />
                            <Typography
                                variant="subtitle2"
                                sx={{
                                    color: '#10B981',
                                    fontWeight: 800,
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.05em',
                                    fontSize: '0.8rem',
                                }}
                            >
                                Critical Evaluation & Empirical Synthesis
                            </Typography>
                        </Box>
                        <Typography variant="body2" sx={{ color: '#F4F4F5', lineHeight: 1.75, fontStyle: 'italic' }}>
                            "{debate.critical_evaluation}"
                        </Typography>
                    </EvaluationCard>
                </DebateContainer>
            ))}
        </Box>
    );
};

export default ConflictingDebates;