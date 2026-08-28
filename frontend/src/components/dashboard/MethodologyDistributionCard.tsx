import React from 'react';
import {
    Box,
    Typography,
    Paper,
    LinearProgress,
    Chip,
} from '@mui/material';
import { styled } from '@mui/system';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import type { MethodologyDistribution } from '../../types';

const CardContainer = styled(Paper)({
    backgroundColor: 'rgba(24, 24, 27, 0.7)',
    backdropFilter: 'blur(20px) saturate(180%)',
    WebkitBackdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '16px',
    padding: '24px',
    marginBottom: '24px',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
    color: '#F4F4F5',
});

const TrendCard = styled(Box)({
    backgroundColor: 'rgba(56, 189, 248, 0.04)',
    border: '1px solid rgba(56, 189, 248, 0.2)',
    borderRadius: '12px',
    padding: '16px 20px',
    marginTop: '20px',
});

const METHOD_COLORS = ['#FFB900', '#00F5C8', '#38BDF8', '#818CF8', '#F43F5E', '#A78BFA', '#34D399'];

interface MethodologyDistributionCardProps {
    overview?: MethodologyDistribution;
    totalPapers?: number;
}

export const MethodologyDistributionCard: React.FC<MethodologyDistributionCardProps> = ({
    overview,
    totalPapers = 0,
}) => {
    if (!overview) return null;

    const distributionEntries = Object.entries(overview.distribution || {});
    const totalCount = distributionEntries.reduce((acc, [, count]) => acc + count, 0) || totalPapers || 1;

    return (
        <CardContainer elevation={0}>
            {/* Header */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, flexWrap: 'wrap', gap: 1.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <AnalyticsIcon sx={{ color: '#FFB900', fontSize: '1.75rem' }} />
                    <Box>
                        <Typography variant="h6" sx={{ fontWeight: 800, color: '#F4F4F5', letterSpacing: '-0.02em' }}>
                            Methodological Landscape & Distribution
                        </Typography>
                        <Typography variant="caption" sx={{ color: '#71717A' }}>
                            Quantitative classification and temporal paradigm shifts across reviewed literature
                        </Typography>
                    </Box>
                </Box>

                {overview.dominant_approach && (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="caption" sx={{ color: '#A1A1AA', fontWeight: 700 }}>
                            Dominant Paradigm:
                        </Typography>
                        <Chip
                            label={overview.dominant_approach}
                            size="small"
                            sx={{
                                backgroundColor: 'rgba(0, 245, 200, 0.15)',
                                color: '#00F5C8',
                                fontWeight: 800,
                                border: '1px solid rgba(0, 245, 200, 0.35)',
                            }}
                        />
                    </Box>
                )}
            </Box>

            {/* Distribution Breakdown Bars */}
            {distributionEntries.length > 0 ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {distributionEntries.map(([method, count], idx) => {
                        const percent = Math.round((count / totalCount) * 100);
                        const color = METHOD_COLORS[idx % METHOD_COLORS.length];

                        return (
                            <Box key={method}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.75 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Box sx={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: color }} />
                                        <Typography variant="body2" sx={{ fontWeight: 700, color: '#E4E4E7' }}>
                                            {method}
                                        </Typography>
                                    </Box>
                                    <Typography variant="caption" sx={{ color: '#A1A1AA', fontWeight: 700 }}>
                                        {count} papers ({percent}%)
                                    </Typography>
                                </Box>
                                <LinearProgress
                                    variant="determinate"
                                    value={percent}
                                    sx={{
                                        height: 8,
                                        borderRadius: 4,
                                        backgroundColor: '#18181B',
                                        '& .MuiLinearProgress-bar': {
                                            borderRadius: 4,
                                            backgroundColor: color,
                                        },
                                    }}
                                />
                            </Box>
                        );
                    })}
                </Box>
            ) : (
                <Typography variant="body2" sx={{ color: '#71717A', fontStyle: 'italic' }}>
                    Methodology distribution data will be computed as papers are analyzed.
                </Typography>
            )}

            {/* Trend Description */}
            {overview.trend_description && (
                <TrendCard>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <TrendingUpIcon sx={{ color: '#38BDF8', fontSize: '1.2rem' }} />
                        <Typography
                            variant="subtitle2"
                            sx={{
                                color: '#38BDF8',
                                fontWeight: 800,
                                textTransform: 'uppercase',
                                letterSpacing: '0.05em',
                                fontSize: '0.78rem',
                            }}
                        >
                            Methodological Evolution & Emerging Trends
                        </Typography>
                    </Box>
                    <Typography variant="body2" sx={{ color: '#D4D4D8', lineHeight: 1.7, fontSize: '0.925rem' }}>
                        {overview.trend_description}
                    </Typography>
                </TrendCard>
            )}
        </CardContainer>
    );
};

export default MethodologyDistributionCard;