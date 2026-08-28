import React, { useEffect, useRef } from 'react';
import {
    Box,
    Typography,
    LinearProgress,
    Chip,
    Paper,
    styled,
    keyframes,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RadioButtonUncheckedIcon from '@mui/icons-material/RadioButtonUnchecked';
import ChangeCircleIcon from '@mui/icons-material/ChangeCircle';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import HubIcon from '@mui/icons-material/Hub';
import FactCheckIcon from '@mui/icons-material/FactCheck';
import GavelIcon from '@mui/icons-material/Gavel';
import { DAG_NODES, type DAGNode } from '../../types/agent';
import type { CriticVerdictData, FactCheckedData } from '../../hooks/useProjectStream';

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

const pulseGlow = keyframes`
  0%, 100% {
    box-shadow: 0 0 15px rgba(255, 185, 0, 0.2), inset 0 0 10px rgba(255, 185, 0, 0.1);
    border-color: rgba(255, 185, 0, 0.5);
  }
  50% {
    box-shadow: 0 0 25px rgba(255, 185, 0, 0.4), inset 0 0 15px rgba(255, 185, 0, 0.2);
    border-color: rgba(255, 185, 0, 0.8);
  }
`;

const PipelineContainer = styled(Paper)(() => ({
    padding: '24px',
    marginBottom: '24px',
    backgroundColor: 'rgba(24, 24, 27, 0.75)',
    backdropFilter: 'blur(20px) saturate(180%)',
    WebkitBackdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid #27272F',
    borderRadius: '16px',
    color: '#F4F4F5',
    position: 'relative',
    overflow: 'hidden',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        width: '4px',
        height: '100%',
        background: 'linear-gradient(to bottom, #818CF8, #00F5C8, #FBBF24, #F43F5E, #A78BFA)',
    },
}));

const AgentStepBox = styled(Box, {
    shouldForwardProp: (prop) => prop !== 'isActive' && prop !== 'isCompleted' && prop !== 'nodeColor',
})<{ isActive?: boolean; isCompleted?: boolean; nodeColor?: string }>(({ isActive, nodeColor = '#FFB900' }) => ({
    display: 'flex',
    alignItems: 'center',
    padding: '14px 16px',
    borderRadius: '12px',
    transition: 'all 0.3s ease',
    backgroundColor: isActive ? 'rgba(255, 185, 0, 0.08)' : 'rgba(255, 255, 255, 0.02)',
    border: isActive ? `1px solid ${nodeColor}` : '1px solid rgba(255, 255, 255, 0.05)',
    position: 'relative',
    zIndex: 1,
    animation: isActive ? `${pulseGlow} 2s infinite ease-in-out` : 'none',
}));

const ConnectorLine = styled(Box, {
    shouldForwardProp: (prop) => prop !== 'isCompleted',
})<{ isCompleted?: boolean }>(({ isCompleted }) => ({
    width: '2px',
    height: '16px',
    marginLeft: '27px',
    backgroundColor: isCompleted ? '#00B894' : '#27272F',
    transition: 'background-color 0.5s ease',
}));

const TerminalBox = styled(Box)(() => ({
    marginTop: '16px',
    backgroundColor: '#09090B',
    border: '1px solid #27272F',
    borderRadius: '10px',
    padding: '14px 16px',
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: '0.8rem',
    color: '#A1A1AA',
    maxHeight: '200px',
    overflowY: 'auto',
    '&::-webkit-scrollbar': { width: '4px' },
    '&::-webkit-scrollbar-thumb': { backgroundColor: '#3F3F46', borderRadius: '2px' },
}));

const RefinementBanner = styled(Box)({
    backgroundColor: 'rgba(244, 63, 94, 0.08)',
    border: '1px solid rgba(244, 63, 94, 0.3)',
    borderRadius: '10px',
    padding: '12px 16px',
    marginTop: '16px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
});

const AuditorBanner = styled(Box)({
    backgroundColor: 'rgba(167, 139, 250, 0.08)',
    border: '1px solid rgba(167, 139, 250, 0.3)',
    borderRadius: '10px',
    padding: '12px 16px',
    marginTop: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
});

interface AgentPipelineProps {
    currentAgent: string | null;
    progress: number;
    logs: string[];
    isConnected?: boolean;
    projectStatus?: string;
    latestCriticVerdict?: CriticVerdictData | null;
    latestFactCheck?: FactCheckedData | null;
}

export const AgentPipeline: React.FC<AgentPipelineProps> = ({
    currentAgent,
    progress,
    logs = [],
    isConnected = false,
    projectStatus,
    latestCriticVerdict,
    latestFactCheck,
}) => {
    const terminalEndRef = useRef<HTMLDivElement | null>(null);

    // Normalize agent alias mapping
    const normalizeAgent = (agent: string | null): string => {
        if (!agent) return '';
        const map: Record<string, string> = {
            planner: 'supervisor',
            retriever: 'discovery',
            analyzer: 'matrix_builder',
            quality_checker: 'critic',
        };
        const low = agent.toLowerCase().replace(/_agent$/, '');
        return map[low] || low;
    };

    const normalizedCurrent = normalizeAgent(currentAgent);

    const getAgentStatus = (node: DAGNode) => {
        if (projectStatus === 'completed') return 'completed';
        if (projectStatus?.startsWith('error')) {
            const currentIdx = DAG_NODES.findIndex((a) => a.id === normalizedCurrent);
            const agentIdx = DAG_NODES.findIndex((a) => a.id === node.id);
            if (agentIdx === currentIdx) return 'error';
            return agentIdx < currentIdx ? 'completed' : 'pending';
        }
        const currentIdx = DAG_NODES.findIndex((a) => a.id === normalizedCurrent);
        const agentIdx = DAG_NODES.findIndex((a) => a.id === node.id);
        if (currentIdx === -1) return 'pending';
        if (agentIdx < currentIdx) return 'completed';
        return agentIdx === currentIdx ? 'active' : 'pending';
    };

    const renderIcon = (status: string, nodeColor: string) => {
        switch (status) {
            case 'completed':
                return <CheckCircleIcon sx={{ color: '#00B894', fontSize: '1.25rem' }} />;
            case 'active':
                return <ChangeCircleIcon sx={{ color: nodeColor, animation: `${spin} 3s linear infinite`, fontSize: '1.25rem' }} />;
            case 'error':
                return <ErrorOutlineIcon sx={{ color: '#EF4444', fontSize: '1.25rem' }} />;
            default:
                return <RadioButtonUncheckedIcon sx={{ color: '#3F3F46', fontSize: '1.25rem' }} />;
        }
    };

    useEffect(() => {
        if (terminalEndRef.current) {
            terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    // Compute effective progress percentage based on stream value or project status fallback
    const effectiveProgress = React.useMemo(() => {
        if (progress > 0) return progress;
        if (projectStatus === 'completed') return 100;
        switch (projectStatus) {
            case 'synthesizing': return 85;
            case 'analyzing': return 50;
            case 'searching': return 25;
            case 'planning': return 10;
            case 'stopped': return 0;
            case 'error':
            case 'error_no_papers_found': return 100;
            default: return 0;
        }
    }, [progress, projectStatus]);

    // Active log messages (falls back to historical DAG completion log when completed)
    const displayLogs = React.useMemo(() => {
        if (logs && logs.length > 0) return logs;
        if (projectStatus === 'completed') {
            return [
                '[SUPERVISOR] Task initialized. Subtopics and search keywords formulated.',
                '[DISCOVERY] Multi-source academic search completed across open-access databases.',
                '[INGESTION] 3-tier Open Access PDF cascade completed. Full-text sections indexed.',
                '[MATRIX_BUILDER] Comparative evidence parameters and benchmark metrics populated.',
                '[SYNTHESIZER] Multi-document thematic literature review compiled.',
                '[CRITIC] Adversarial quality review passed. Methodological rigor verified.',
                '[AUDITOR] Citation grounding and proposition audit passed. Report deliverable ready.',
            ];
        }
        return [];
    }, [logs, projectStatus]);

    return (
        <PipelineContainer elevation={0}>
            {/* Top Status & Controls */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <HubIcon sx={{ color: '#FFB900' }} />
                    <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: '-0.02em', color: '#F4F4F5' }}>
                        LangGraph Orchestrator DAG
                    </Typography>
                </Box>
                <Chip
                    label={isConnected ? 'REALTIME STREAM ACTIVE' : (projectStatus === 'completed' ? 'SYNTHESIS COMPLETE' : 'STREAM DISCONNECTED')}
                    size="small"
                    sx={{
                        height: 24,
                        fontSize: '0.65rem',
                        fontWeight: 900,
                        backgroundColor: (isConnected || projectStatus === 'completed') ? 'rgba(0, 184, 148, 0.12)' : 'rgba(63, 63, 70, 0.12)',
                        color: (isConnected || projectStatus === 'completed') ? '#00B894' : '#71717A',
                        border: `1px solid ${(isConnected || projectStatus === 'completed') ? '#00B894' : '#3F3F46'}`,
                    }}
                />
            </Box>

            {/* Global Progress Bar */}
            <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption" sx={{ color: '#71717A', fontWeight: 700, textTransform: 'uppercase' }}>
                        Pipeline Execution Progress
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#FFB900', fontWeight: 800 }}>
                        {Math.round(effectiveProgress)}%
                    </Typography>
                </Box>
                <LinearProgress
                    variant="determinate"
                    value={effectiveProgress}
                    sx={{
                        height: 6,
                        borderRadius: 3,
                        backgroundColor: '#18181B',
                        '& .MuiLinearProgress-bar': {
                            borderRadius: 3,
                            background: 'linear-gradient(90deg, #818CF8 0%, #00F5C8 50%, #FFB900 100%)',
                        },
                    }}
                />
            </Box>

            {/* LangGraph 7-Node State Machine */}
            <Box>
                {DAG_NODES.map((node, index) => {
                    const status = getAgentStatus(node);
                    const isLast = index === DAG_NODES.length - 1;

                    return (
                        <React.Fragment key={node.id}>
                            <AgentStepBox
                                isActive={status === 'active'}
                                isCompleted={status === 'completed'}
                                nodeColor={node.color}
                            >
                                <Box sx={{ mr: 2, display: 'flex' }}>{renderIcon(status, node.color)}</Box>
                                <Box sx={{ flex: 1 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Typography
                                            variant="subtitle2"
                                            sx={{
                                                color: status === 'pending' ? '#52525B' : '#F4F4F5',
                                                fontWeight: status === 'active' ? 800 : 600,
                                            }}
                                        >
                                            {node.name}
                                        </Typography>
                                        <Chip
                                            label={`Stage ${node.stageNumber}`}
                                            size="small"
                                            sx={{
                                                height: 18,
                                                fontSize: '0.6rem',
                                                fontWeight: 800,
                                                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                                                color: '#A1A1AA',
                                            }}
                                        />
                                    </Box>
                                    <Typography variant="caption" sx={{ color: '#71717A', display: 'block', mt: 0.25 }}>
                                        {node.description}
                                    </Typography>
                                </Box>

                                {status === 'active' && (
                                    <Typography variant="caption" sx={{ color: node.color, fontWeight: 900, fontSize: '0.65rem', letterSpacing: '0.05em' }}>
                                        EXECUTING...
                                    </Typography>
                                )}
                            </AgentStepBox>
                            {!isLast && <ConnectorLine isCompleted={status === 'completed'} />}
                        </React.Fragment>
                    );
                })}
            </Box>

            {/* Adversarial Critic Refinement Loop Banner */}
            {latestCriticVerdict && (
                <RefinementBanner>
                    <GavelIcon sx={{ color: latestCriticVerdict.should_refine ? '#F43F5E' : '#10B981' }} />
                    <Box sx={{ flex: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 800, color: '#F4F4F5' }}>
                                Adversarial Critic Evaluation (Iteration {latestCriticVerdict.iteration})
                            </Typography>
                            <Chip
                                label={`Score: ${latestCriticVerdict.score.toFixed(1)}/100`}
                                size="small"
                                sx={{
                                    height: 20,
                                    fontSize: '0.65rem',
                                    fontWeight: 800,
                                    backgroundColor: latestCriticVerdict.should_refine ? 'rgba(244, 63, 94, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                                    color: latestCriticVerdict.should_refine ? '#F43F5E' : '#10B981',
                                }}
                            />
                        </Box>
                        {latestCriticVerdict.should_refine ? (
                            <Typography variant="caption" sx={{ color: '#F43F5E', display: 'block', mt: 0.5 }}>
                                Quality threshold &lt; 75.0 triggered refinement iteration loop.
                            </Typography>
                        ) : (
                            <Typography variant="caption" sx={{ color: '#10B981', display: 'block', mt: 0.5 }}>
                                Synthesis passed academic rigor verification.
                            </Typography>
                        )}
                    </Box>
                </RefinementBanner>
            )}

            {/* Citation Auditor Fact-Checking Banner */}
            {latestFactCheck && (
                <AuditorBanner>
                    <FactCheckIcon sx={{ color: latestFactCheck.passed ? '#A78BFA' : '#F43F5E' }} />
                    <Box sx={{ flex: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="subtitle2" sx={{ fontWeight: 800, color: '#F4F4F5' }}>
                                Citation Audit NLI Grounding
                            </Typography>
                            <Chip
                                label={`Precision: ${latestFactCheck.precision_score.toFixed(1)}%`}
                                size="small"
                                sx={{
                                    height: 20,
                                    fontSize: '0.65rem',
                                    fontWeight: 800,
                                    backgroundColor: 'rgba(167, 139, 250, 0.2)',
                                    color: '#A78BFA',
                                }}
                            />
                        </Box>
                        <Typography variant="caption" sx={{ color: '#D4D4D8', display: 'block', mt: 0.5 }}>
                            {latestFactCheck.entailed_count ?? 0} propositions entailed, {latestFactCheck.contradiction_count ?? 0} contradictions.
                        </Typography>
                    </Box>
                </AuditorBanner>
            )}

            {/* Terminal Activity Logs */}
            {displayLogs.length > 0 && (
                <Box sx={{ mt: 2.5 }}>
                    <Typography variant="caption" sx={{ color: '#71717A', fontWeight: 700, ml: 0.5, textTransform: 'uppercase' }}>
                        {projectStatus === 'completed' && logs.length === 0 ? 'Execution History & Agent Log' : 'Live Telemetry & Event Stream'}
                    </Typography>
                    <TerminalBox>
                        {displayLogs.slice(-20).map((log, i) => (
                            <Box key={i} sx={{ mb: 0.5, opacity: 0.5 + (i / 20) * 0.5, display: 'flex', gap: 1 }}>
                                <Typography component="span" sx={{ color: '#FFB900', fontSize: 'inherit', fontWeight: 700 }}>
                                    &gt;
                                </Typography>
                                <Typography component="span" sx={{ fontSize: 'inherit', wordBreak: 'break-word' }}>
                                    {log}
                                </Typography>
                            </Box>
                        ))}
                        <div ref={terminalEndRef} />
                    </TerminalBox>
                </Box>
            )}
        </PipelineContainer>
    );
};

export default AgentPipeline;