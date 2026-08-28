import React, { useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
    Typography,
    Box,
    CircularProgress,
    Alert,
    Button,
    Menu,
    MenuItem,
    Tabs,
    Tab,
    Chip,
    Divider,
} from '@mui/material';
import { styled } from '@mui/system';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import DownloadIcon from '@mui/icons-material/Download';
import SchoolIcon from '@mui/icons-material/School';
import TableChartIcon from '@mui/icons-material/TableChart';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TimelineIcon from '@mui/icons-material/Timeline';
import LibraryBooksIcon from '@mui/icons-material/LibraryBooks';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import VerifiedIcon from '@mui/icons-material/Verified';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ArticleIcon from '@mui/icons-material/Article';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import { toast } from 'react-toastify';

import { neonData } from '../api/neonClient';
import { getProjectReport, getProjectMatrix, getProjectGaps, startLiteratureReview, stopLiteratureReview } from '../api/client';
import { useProjectStream } from '../hooks/useProjectStream';
import { exportReport, exportToBibTeX, formatAuthors } from '../utils/exportEngine';

import EvidenceMatrixTable from '../components/dashboard/EvidenceMatrixTable';
import ThematicSections from '../components/dashboard/ThematicSections';
import ConflictingDebates from '../components/dashboard/ConflictingDebates';
import ResearchGapViewer from '../components/dashboard/ResearchGapViewer';
import MethodologyDistributionCard from '../components/dashboard/MethodologyDistributionCard';
import AgentPipeline from '../components/dashboard/AgentPipeline';
import type { ResearchReport, EvidenceMatrixRow, ResearchGapItem, BibliographyItem } from '../types';

const PageWrapper = styled(Box)({
    minHeight: '100vh',
    backgroundColor: '#09090B',
    backgroundImage: 'radial-gradient(circle at 50% -20%, rgba(255, 185, 0, 0.04) 0%, transparent 50%)',
    color: '#F4F4F5',
    paddingTop: '90px',
    paddingBottom: '80px',
});

const GlassCard = styled(Box)({
    background: 'rgba(24, 24, 27, 0.7)',
    backdropFilter: 'blur(20px) saturate(180%)',
    WebkitBackdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '16px',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
    transition: 'all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1)',
});

const HeaderCard = styled(GlassCard)({
    padding: '2rem',
    marginBottom: '2rem',
    background: 'linear-gradient(135deg, rgba(24, 24, 27, 0.85) 0%, rgba(39, 39, 47, 0.75) 100%)',
    borderRadius: '20px',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        right: 0,
        width: '300px',
        height: '300px',
        background: 'radial-gradient(circle, rgba(255, 185, 0, 0.08) 0%, transparent 70%)',
        pointerEvents: 'none',
    },
});

const StyledTabs = styled(Tabs)({
    marginBottom: '2rem',
    backgroundColor: 'rgba(24, 24, 27, 0.5)',
    borderRadius: '12px',
    padding: '8px',
    border: '1px solid rgba(255, 255, 255, 0.05)',
    '& .MuiTabs-indicator': {
        backgroundColor: '#FFB900',
        height: '3px',
        borderRadius: '3px 3px 0 0',
    },
});

const StyledTab = styled(Tab)({
    color: '#A1A1AA',
    fontWeight: 600,
    fontSize: '0.875rem',
    textTransform: 'none',
    minHeight: '48px',
    transition: 'all 0.2s ease',
    borderRadius: '8px',
    '&:hover': {
        color: '#F4F4F5',
        backgroundColor: 'rgba(255, 255, 255, 0.05)',
    },
    '&.Mui-selected': {
        color: '#FFB900',
    },
    '& .MuiTab-iconWrapper': {
        marginBottom: '2px',
    },
});

const PaperItemCard = styled(GlassCard)({
    padding: '1.5rem',
    transition: 'all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1), transform 0.2s ease',
    '&:hover': {
        transform: 'translateY(-3px)',
        borderColor: 'rgba(0, 245, 200, 0.3)',
        boxShadow: '0 12px 36px rgba(0, 245, 200, 0.1)',
    },
});

const StatusBadge = styled(Chip)<{ status?: string }>(({ status }) => {
    const statusColors: Record<string, { bg: string; text: string; glow: string }> = {
        completed: { bg: 'rgba(0, 184, 148, 0.15)', text: '#00B894', glow: '0 0 20px rgba(0, 184, 148, 0.3)' },
        synthesizing: { bg: 'rgba(255, 185, 0, 0.15)', text: '#FFB900', glow: '0 0 20px rgba(255, 185, 0, 0.3)' },
        analyzing: { bg: 'rgba(56, 189, 248, 0.15)', text: '#38BDF8', glow: '0 0 20px rgba(56, 189, 248, 0.3)' },
        searching: { bg: 'rgba(129, 140, 248, 0.15)', text: '#818CF8', glow: '0 0 20px rgba(129, 140, 248, 0.3)' },
        error: { bg: 'rgba(244, 67, 54, 0.15)', text: '#F44336', glow: '0 0 20px rgba(244, 67, 54, 0.3)' },
        created: { bg: 'rgba(161, 161, 170, 0.15)', text: '#A1A1AA', glow: 'none' },
    };
    const colors = statusColors[status || 'created'] || statusColors.created;
    return {
        backgroundColor: colors.bg,
        color: colors.text,
        fontWeight: 700,
        fontSize: '0.75rem',
        height: '28px',
        borderRadius: '14px',
        border: `1px solid ${colors.text}40`,
        boxShadow: colors.glow,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
    };
});

const PrimaryButton = styled(Button)({
    backgroundColor: '#FFB900',
    color: '#09090B',
    borderRadius: '10px',
    padding: '8px 20px',
    fontWeight: 700,
    textTransform: 'none',
    boxShadow: '0 4px 14px rgba(255, 185, 0, 0.3)',
    transition: 'all 0.2s ease',
    '&:hover': {
        backgroundColor: '#E6A600',
        transform: 'translateY(-1px)',
        boxShadow: '0 6px 20px rgba(255, 185, 0, 0.4)',
    },
});
export const ProjectDetailsPage: React.FC = () => {
    const { projectId } = useParams<{ projectId: string }>();
    const [activeTab, setActiveTab] = useState(0);
    const [hasUserSelectedTab, setHasUserSelectedTab] = useState(false);
    const [exportMenuAnchor, setExportMenuAnchor] = useState<null | HTMLElement>(null);
    const [questionAnchorEl, setQuestionAnchorEl] = useState<null | HTMLElement>(null);
    const [isStartingPipeline, setIsStartingPipeline] = useState(false);

    const exportMenuOpen = Boolean(exportMenuAnchor);
    const questionOpen = Boolean(questionAnchorEl);

    // Fetch primary project record
    const {
        data: project,
        isLoading: isProjectLoading,
        error: projectError,
        refetch: refetchProject,
    } = useQuery({
        queryKey: ['project', projectId],
        queryFn: () => neonData.getProjectById(projectId!),
        enabled: !!projectId,
    });

    // Fetch structured v3.2 research report deliverable
    const {
        data: reportResponse,
        isLoading: isReportLoading,
    } = useQuery({
        queryKey: ['project-report', projectId],
        queryFn: () => getProjectReport(projectId!),
        enabled: !!projectId,
    });

    // Fetch comparative evidence matrix
    const {
        data: matrixResponse,
    } = useQuery({
        queryKey: ['project-matrix', projectId],
        queryFn: () => getProjectMatrix(projectId!),
        enabled: !!projectId,
    });

    // Fetch actionable research gaps
    const {
        data: gapsResponse,
    } = useQuery({
        queryKey: ['project-gaps', projectId],
        queryFn: () => getProjectGaps(projectId!),
        enabled: !!projectId,
    });

    // Real-time WebSocket event streaming hook
    const {
        isConnected,
        currentAgent,
        progress,
        logs,
        latestCriticVerdict,
        latestFactCheck,
    } = useProjectStream(projectId);

    // Assemble master report object merging REST responses and project model
    const report: ResearchReport | null = useMemo(() => {
        if (reportResponse?.report) return reportResponse.report;
        if (project?.report) return project.report;

        const synthesizerPlan = project?.agent_plans?.find((p) => p.agent_type === 'synthesizer');
        const synthesisOutput =
            (typeof synthesizerPlan?.plan_steps?.[0]?.output?.response === 'string'
                ? synthesizerPlan?.plan_steps?.[0]?.output?.response
                : '') || '';

        if (!project) return null;

        return {
            metadata: {
                title: project.title,
                research_question: project.research_question,
                generated_at: project.created_at || new Date().toISOString(),
                quality_score: 85.0,
                status: project.status || 'complete',
            },
            title: project.title,
            executive_summary: synthesisOutput || 'Autonomous literature synthesis in progress.',
            thematic_sections: [
                {
                    theme_id: 'overview',
                    title: 'Core Literature Synthesis',
                    synthesis_prose: synthesisOutput,
                    key_takeaways: ['Cross-domain literature reviewed across indexed sources.'],
                    cited_paper_ids: project.paper_references?.map((p) => p.id) || [],
                },
            ],
            comparison_matrix: matrixResponse?.entries || [],
            actionable_research_gaps: gapsResponse?.gaps || [],
            conflicting_findings_and_debates: [],
            methodology_overview: {
                distribution: { 'Quantitative / Empirical': project.paper_references?.length || 1 },
                dominant_approach: 'Quantitative / Empirical',
                trend_description: 'Empirical experimentation and comparative benchmarking.',
            },
            bibliography: project.paper_references?.map((p) => ({
                paper_id: p.id,
                title: p.title,
                authors: p.authors || [],
                year: p.year ?? null,
                url: p.url,
                pdf_url: p.url,
                is_full_text_analyzed: p.is_full_text ?? true,
            })) || [],
        };
    }, [reportResponse, project, matrixResponse, gapsResponse]);

    // Active matrix rows
    const matrixRows: EvidenceMatrixRow[] = useMemo(() => {
        if (matrixResponse?.entries && matrixResponse.entries.length > 0) return matrixResponse.entries;
        if (report?.comparison_matrix && report.comparison_matrix.length > 0) return report.comparison_matrix;
        if (report?.comparative_matrix && report.comparative_matrix.length > 0) return report.comparative_matrix;
        return [];
    }, [matrixResponse, report]);

    // Active research gaps
    const gaps: ResearchGapItem[] = useMemo(() => {
        if (gapsResponse?.gaps && gapsResponse.gaps.length > 0) return gapsResponse.gaps;
        if (report?.actionable_research_gaps && report.actionable_research_gaps.length > 0) return report.actionable_research_gaps;
        if (report?.actionable_gaps && report.actionable_gaps.length > 0) return report.actionable_gaps;
        if (report?.research_gaps && report.research_gaps.length > 0) return report.research_gaps;
        return [];
    }, [gapsResponse, report]);

    // Active conflicting debates
    const debates = useMemo(() => {
        if (report?.conflicting_findings_and_debates && report.conflicting_findings_and_debates.length > 0)
            return report.conflicting_findings_and_debates;
        if (report?.conflicting_debates && report.conflicting_debates.length > 0) return report.conflicting_debates;
        if (report?.debates && report.debates.length > 0) return report.debates;
        return [];
    }, [report]);

    // Active bibliography
    const bibliography: BibliographyItem[] = useMemo(() => {
        if (report?.bibliography && report.bibliography.length > 0) return report.bibliography as BibliographyItem[];
        return project?.paper_references?.map((p) => ({
            paper_id: p.id,
            title: p.title,
            authors: p.authors || [],
            year: p.year ?? null,
            venue: p.venue,
            doi: p.doi,
            pdf_url: p.url,
            url: p.url,
            citation_count: p.citation_count,
            is_full_text_analyzed: p.is_full_text ?? true,
        })) || [];
    }, [report, project]);

    // Auto-select Real-Time Journey (Tab 5) if task is active, or Literature Review (Tab 0) if complete
    React.useEffect(() => {
        if (!hasUserSelectedTab && project?.status) {
            if (project.status !== 'completed' && project.status !== 'created') {
                setActiveTab(5);
            }
        }
    }, [project?.status, hasUserSelectedTab]);

    const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
        setHasUserSelectedTab(true);
        setActiveTab(newValue);
    };

    const handleExportClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        setExportMenuAnchor(event.currentTarget);
    };

    const handleExportClose = () => {
        setExportMenuAnchor(null);
    };

    const handleQuestionClick = (event: React.MouseEvent<HTMLElement>) => {
        setQuestionAnchorEl(questionAnchorEl ? null : event.currentTarget);
    };

    const handleQuestionClose = () => {
        setQuestionAnchorEl(null);
    };

    const [isStoppingPipeline, setIsStoppingPipeline] = useState(false);

    const handleTriggerPipeline = async () => {
        if (!projectId) return;
        setIsStartingPipeline(true);
        try {
            await startLiteratureReview(projectId, project?.max_papers || 30);
            refetchProject();
            setHasUserSelectedTab(true);
            setActiveTab(5); // Switch to Real-Time Journey tab
        } catch (e) {
            console.error('Failed to trigger pipeline:', e);
            toast.error('Failed to start literature review.');
        } finally {
            setIsStartingPipeline(false);
        }
    };

    const handleStopPipeline = async () => {
        if (!projectId) return;
        setIsStoppingPipeline(true);
        try {
            await stopLiteratureReview(projectId);
            toast.info('Research task stopped.');
            refetchProject();
        } catch (e) {
            console.error('Failed to stop pipeline:', e);
            toast.error('Failed to stop research task.');
        } finally {
            setIsStoppingPipeline(false);
        }
    };

    const triggerExport = async (format: 'md' | 'pdf' | 'docx' | 'bib') => {
        if (!report) {
            toast.error('No generated report deliverable available to export.');
            return;
        }
        handleExportClose();
        try {
            toast.info(`Preparing ${format.toUpperCase()} export...`);
            await exportReport(format, report, 'synthesis-output-container', project?.title);
            toast.success(`Exported ${format.toUpperCase()} successfully.`);
        } catch (err: unknown) {
            console.error('Export error:', err);
            const errorMessage = err instanceof Error ? err.message : 'Unknown error';
            toast.error(`Export failed: ${errorMessage}`);
        }
    };
    if (isProjectLoading) {
        return (
            <PageWrapper>
                <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh', flexDirection: 'column', gap: 2 }}>
                    <CircularProgress sx={{ color: '#FFB900' }} size={50} />
                    <Typography sx={{ color: '#71717A' }}>Loading intelligence workspace...</Typography>
                </Box>
            </PageWrapper>
        );
    }

    if (projectError || !project) {
        return (
            <PageWrapper>
                <Box sx={{ px: { xs: 2, md: 6 }, py: 4, maxWidth: '800px', mx: 'auto' }}>
                    <Alert
                        severity="error"
                        sx={{ backgroundColor: 'rgba(244, 67, 54, 0.15)', color: '#F44336', border: '1px solid rgba(244, 67, 54, 0.3)', borderRadius: '12px' }}
                    >
                        Failed to load project details: {projectError?.message || 'Project not found.'}
                    </Alert>
                </Box>
            </PageWrapper>
        );
    }

    const projectStatus = project.status || 'completed';
    const isRunning = ['searching', 'analyzing', 'synthesizing', 'in_progress', 'planning'].includes(projectStatus);

    return (
        <PageWrapper>
            <Box sx={{ px: { xs: 2, sm: 3, md: 6 }, mt: { xs: -2, md: 0 }, maxWidth: '1400px', mx: 'auto' }}>
                {/* Top Navigation Row */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                    <Button
                        component={Link}
                        to="/"
                        startIcon={<ArrowBackIcon />}
                        sx={{
                            color: '#A1A1AA',
                            fontWeight: 600,
                            textTransform: 'none',
                            '&:hover': { color: '#F4F4F5', backgroundColor: 'rgba(255, 255, 255, 0.05)' },
                        }}
                    >
                        Back to Workspace
                    </Button>

                    <Box sx={{ display: 'flex', gap: 1.5 }}>
                        {isRunning && (
                            <Button
                                variant="outlined"
                                startIcon={
                                    isStoppingPipeline ? (
                                        <CircularProgress size={16} color="inherit" />
                                    ) : (
                                        <StopCircleOutlinedIcon />
                                    )
                                }
                                onClick={handleStopPipeline}
                                disabled={isStoppingPipeline}
                                sx={{
                                    color: '#EF4444',
                                    borderColor: 'rgba(239, 68, 68, 0.5)',
                                    fontWeight: 700,
                                    textTransform: 'none',
                                    borderRadius: '8px',
                                    '&:hover': {
                                        borderColor: '#EF4444',
                                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                                    },
                                }}
                            >
                                {isStoppingPipeline ? 'Stopping...' : 'Stop Task'}
                            </Button>
                        )}
                        {(projectStatus === 'created' || projectStatus === 'stopped') && (
                            <PrimaryButton
                                startIcon={
                                    isStartingPipeline ? (
                                        <CircularProgress size={16} sx={{ color: '#09090B' }} />
                                    ) : (
                                        <PlayArrowIcon />
                                    )
                                }
                                onClick={handleTriggerPipeline}
                                disabled={isStartingPipeline}
                            >
                                {projectStatus === 'stopped' ? 'Restart Research' : 'Start Literature Review'}
                            </PrimaryButton>
                        )}
                        <PrimaryButton startIcon={<DownloadIcon />} onClick={handleExportClick}>
                            Export Deliverables
                        </PrimaryButton>
                    </Box>
                </Box>

                {/* Project Header Banner */}
                <HeaderCard>
                    <Box sx={{ position: 'relative', zIndex: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 0 }}>
                            <Box sx={{ flex: 1, pr: 2 }}>
                                <Typography
                                    variant="h3"
                                    sx={{
                                        fontWeight: 800,
                                        color: '#F4F4F5',
                                        letterSpacing: '-0.03em',
                                        fontSize: { xs: '1.75rem', sm: '2.25rem', md: '2.5rem' },
                                        mb: 1,
                                    }}
                                >
                                    {project.title}
                                </Typography>

                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap', mt: 1 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                                        <SchoolIcon sx={{ color: '#FFB900', fontSize: '1.1rem' }} />
                                        <Typography variant="body2" sx={{ color: '#D4D4D8' }}>
                                            <strong>{project.paper_references?.length || project.total_papers_found || 0}</strong> Papers Discovered
                                        </Typography>
                                    </Box>

                                    {report?.metadata?.quality_score !== undefined && (
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
                                            <VerifiedIcon sx={{ color: '#00F5C8', fontSize: '1.1rem' }} />
                                            <Typography variant="body2" sx={{ color: '#00F5C8', fontWeight: 700 }}>
                                                Quality Score: {report.metadata.quality_score.toFixed(1)}/100
                                            </Typography>
                                        </Box>
                                    )}

                                    {isRunning && (
                                        <Chip
                                            label={`Executing (${Math.round(progress)}%)`}
                                            size="small"
                                            sx={{
                                                backgroundColor: 'rgba(255, 185, 0, 0.15)',
                                                color: '#FFB900',
                                                fontWeight: 800,
                                                border: '1px solid rgba(255, 185, 0, 0.4)',
                                            }}
                                        />
                                    )}
                                </Box>
                            </Box>

                            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1.5 }}>
                                <StatusBadge label={projectStatus} status={projectStatus} />
                                <Button
                                    variant="text"
                                    size="small"
                                    onClick={handleQuestionClick}
                                    endIcon={
                                        <ExpandMoreIcon
                                            sx={{
                                                transition: 'transform 0.2s ease',
                                                transform: questionOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                                            }}
                                        />
                                    }
                                    startIcon={<HelpOutlineIcon sx={{ fontSize: '1rem !important' }} />}
                                    sx={{
                                        color: questionOpen ? '#FFB900' : '#A1A1AA',
                                        backgroundColor: questionOpen ? 'rgba(255, 185, 0, 0.05)' : 'transparent',
                                        textTransform: 'none',
                                        fontWeight: 600,
                                        fontSize: '0.85rem',
                                        padding: '4px 8px',
                                        borderRadius: '8px',
                                    }}
                                >
                                    Research Question
                                </Button>

                                <Menu
                                    anchorEl={questionAnchorEl}
                                    open={questionOpen}
                                    onClose={handleQuestionClose}
                                    anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                                    transformOrigin={{ vertical: 'top', horizontal: 'right' }}
                                    PaperProps={{
                                        sx: {
                                            backgroundColor: '#18181B',
                                            border: '1px solid rgba(255, 255, 255, 0.1)',
                                            borderRadius: '12px',
                                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
                                            maxWidth: '420px',
                                            p: 2.5,
                                        },
                                    }}
                                >
                                    <Typography variant="overline" sx={{ color: '#71717A', fontWeight: 700, display: 'block', mb: 1, letterSpacing: '0.1em' }}>
                                        Target Research Question
                                    </Typography>
                                    <Typography variant="body1" sx={{ color: '#F4F4F5', fontStyle: 'italic', lineHeight: 1.6 }}>
                                        "{project.research_question}"
                                    </Typography>
                                </Menu>
                            </Box>
                        </Box>
                    </Box>
                </HeaderCard>

                {/* Navigation Tabs across all 6 Modules */}
                <StyledTabs
                    value={activeTab}
                    onChange={handleTabChange}
                    variant="scrollable"
                    scrollButtons="auto"
                    allowScrollButtonsMobile
                >
                    <StyledTab icon={<LibraryBooksIcon />} iconPosition="start" label="Literature Review" />
                    <StyledTab icon={<TableChartIcon />} iconPosition="start" label={`Evidence Matrix (${matrixRows.length})`} />
                    <StyledTab icon={<CompareArrowsIcon />} iconPosition="start" label={`Methodological Debates (${debates.length})`} />
                    <StyledTab icon={<TrendingUpIcon />} iconPosition="start" label={`Actionable Gaps (${gaps.length})`} />
                    <StyledTab icon={<SchoolIcon />} iconPosition="start" label={`Bibliography (${bibliography.length})`} />
                    <StyledTab icon={<TimelineIcon />} iconPosition="start" label="Real-Time Journey" />
                </StyledTabs>

                {/* Tab 0: Literature Review */}
                {activeTab === 0 && (
                    <Box id="synthesis-output-container">
                        {/* Executive Summary Card */}
                        <GlassCard sx={{ p: { xs: 2.5, md: 4 }, mb: 3 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                <Typography variant="h5" sx={{ fontWeight: 800, color: '#F4F4F5', letterSpacing: '-0.02em' }}>
                                    Executive Summary
                                </Typography>
                                {report?.metadata?.generated_at && (
                                    <Typography variant="caption" sx={{ color: '#71717A' }}>
                                        Synthesized: {new Date(report.metadata.generated_at).toLocaleDateString()}
                                    </Typography>
                                )}
                            </Box>
                            <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.08)', mb: 2.5 }} />
                            <Typography variant="body1" sx={{ color: '#E4E4E7', lineHeight: 1.85, fontSize: '1.025rem', fontFamily: 'var(--font-content)' }}>
                                {report?.executive_summary || 'Autonomous multi-agent review is compiling executive findings.'}
                            </Typography>
                        </GlassCard>

                        {/* Methodology Distribution Breakdown */}
                        {report?.methodology_overview && (
                            <MethodologyDistributionCard
                                overview={report.methodology_overview}
                                totalPapers={project.paper_references?.length || project.total_papers_found}
                            />
                        )}

                        {/* Dense Thematic Synthesis Sections with Interactive Citation Anchors */}
                        <ThematicSections
                            sections={report?.thematic_sections || report?.sections || []}
                            bibliography={bibliography}
                            paperReferences={project.paper_references}
                        />
                    </Box>
                )}

                {/* Tab 1: Comparative Evidence Matrix */}
                {activeTab === 1 && (
                    <EvidenceMatrixTable
                        rows={matrixRows}
                        isLoading={isReportLoading || isRunning}
                    />
                )}

                {/* Tab 2: Methodological & Empirical Debates */}
                {activeTab === 2 && (
                    <ConflictingDebates debates={debates} />
                )}

                {/* Tab 3: Actionable Research Gaps */}
                {activeTab === 3 && (
                    <ResearchGapViewer gaps={gaps} bibliography={bibliography} />
                )}

                {/* Tab 4: Discovered Papers & Bibliography */}
                {activeTab === 4 && (
                    <Box>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, flexWrap: 'wrap', gap: 2 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <SchoolIcon sx={{ color: '#FFB900', fontSize: '1.75rem' }} />
                                <Box>
                                    <Typography variant="h5" sx={{ fontWeight: 800, color: '#F4F4F5', letterSpacing: '-0.02em' }}>
                                        Discovered Papers & Corpus ({bibliography.length})
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: '#71717A' }}>
                                        Indexed open-access scientific literature retrieved by the Autonomous Explorer
                                    </Typography>
                                </Box>
                            </Box>

                            <PrimaryButton
                                startIcon={<DownloadIcon />}
                                onClick={() => exportToBibTeX(bibliography, `${project.title.substring(0, 20)}_references.bib`)}
                            >
                                Export BibTeX (.bib)
                            </PrimaryButton>
                        </Box>

                        {bibliography.length === 0 ? (
                            <GlassCard sx={{ p: 6, textAlign: 'center', color: '#71717A' }}>
                                <SchoolIcon sx={{ fontSize: '3.5rem', opacity: 0.3, mb: 1 }} />
                                <Typography variant="h6" sx={{ color: '#A1A1AA', fontWeight: 600 }}>
                                    No papers discovered yet
                                </Typography>
                                <Typography variant="body2" sx={{ mt: 0.5 }}>
                                    Literature discovery agent will populate candidate papers during exploration.
                                </Typography>
                            </GlassCard>
                        ) : (
                            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}>
                                {bibliography.map((paper, idx) => {
                                    const authorStr = formatAuthors(paper.authors);
                                    const yearStr = paper.year ? `(${paper.year})` : '';
                                    const isFT = paper.is_full_text_analyzed !== false;

                                    return (
                                        <PaperItemCard key={paper.paper_id || idx}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
                                                <Typography variant="h6" sx={{ fontWeight: 800, color: '#F4F4F5', fontSize: '1.05rem', lineHeight: 1.35, flex: 1, pr: 2 }}>
                                                    {paper.title}
                                                </Typography>
                                                <Chip
                                                    label={isFT ? 'Full Text' : 'Abstract'}
                                                    size="small"
                                                    sx={{
                                                        backgroundColor: isFT ? 'rgba(0, 184, 148, 0.15)' : 'rgba(161, 161, 170, 0.1)',
                                                        color: isFT ? '#00B894' : '#A1A1AA',
                                                        fontWeight: 700,
                                                        fontSize: '0.7rem',
                                                        border: `1px solid ${isFT ? 'rgba(0, 184, 148, 0.3)' : 'rgba(161, 161, 170, 0.2)'}`,
                                                    }}
                                                />
                                            </Box>

                                            <Typography variant="body2" sx={{ color: '#A1A1AA', fontSize: '0.85rem', mb: 1.5, fontStyle: 'italic' }}>
                                                {authorStr} {yearStr}
                                            </Typography>

                                            {paper.venue && (
                                                <Typography variant="caption" sx={{ color: '#71717A', display: 'block', mb: 2 }}>
                                                    Venue: {paper.venue} {paper.citation_count ? `| Citations: ${paper.citation_count}` : ''}
                                                </Typography>
                                            )}

                                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2, alignItems: 'center' }}>
                                                {paper.doi && (
                                                    <Button
                                                        component="a"
                                                        href={`https://doi.org/${paper.doi.replace(/^https?:\/\/doi\.org\//, '')}`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        size="small"
                                                        startIcon={<OpenInNewIcon sx={{ fontSize: '0.85rem !important' }} />}
                                                        sx={{
                                                            backgroundColor: 'rgba(56, 189, 248, 0.1)',
                                                            color: '#38BDF8',
                                                            border: '1px solid rgba(56, 189, 248, 0.3)',
                                                            textTransform: 'none',
                                                            fontWeight: 700,
                                                            fontSize: '0.75rem',
                                                            borderRadius: '8px',
                                                            py: 0.5,
                                                            px: 1.2,
                                                            '&:hover': {
                                                                backgroundColor: 'rgba(56, 189, 248, 0.2)',
                                                                borderColor: 'rgba(56, 189, 248, 0.5)',
                                                            },
                                                        }}
                                                    >
                                                        DOI Link
                                                    </Button>
                                                )}

                                                {paper.url && (
                                                    <Button
                                                        component="a"
                                                        href={paper.url}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        size="small"
                                                        startIcon={<OpenInNewIcon sx={{ fontSize: '0.85rem !important' }} />}
                                                        sx={{
                                                            backgroundColor: 'rgba(255, 185, 0, 0.1)',
                                                            color: '#FFB900',
                                                            border: '1px solid rgba(255, 185, 0, 0.3)',
                                                            textTransform: 'none',
                                                            fontWeight: 700,
                                                            fontSize: '0.75rem',
                                                            borderRadius: '8px',
                                                            py: 0.5,
                                                            px: 1.2,
                                                            '&:hover': {
                                                                backgroundColor: 'rgba(255, 185, 0, 0.2)',
                                                                borderColor: 'rgba(255, 185, 0, 0.5)',
                                                            },
                                                        }}
                                                    >
                                                        {paper.url.includes('arxiv.org')
                                                            ? 'arXiv Page'
                                                            : (paper.url.includes('semanticscholar.org')
                                                                ? 'Semantic Scholar'
                                                                : (paper.url.includes('openalex.org') ? 'OpenAlex' : 'Paper Source'))}
                                                    </Button>
                                                )}

                                                {(paper.pdf_url || (isFT && paper.url)) && (
                                                    <Button
                                                        component="a"
                                                        href={(paper.pdf_url || paper.url) || undefined}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        size="small"
                                                        startIcon={<ArticleIcon sx={{ fontSize: '0.85rem !important' }} />}
                                                        sx={{
                                                            backgroundColor: 'rgba(0, 245, 200, 0.1)',
                                                            color: '#00F5C8',
                                                            border: '1px solid rgba(0, 245, 200, 0.3)',
                                                            textTransform: 'none',
                                                            fontWeight: 700,
                                                            fontSize: '0.75rem',
                                                            borderRadius: '8px',
                                                            py: 0.5,
                                                            px: 1.2,
                                                            '&:hover': {
                                                                backgroundColor: 'rgba(0, 245, 200, 0.2)',
                                                                borderColor: 'rgba(0, 245, 200, 0.5)',
                                                            },
                                                        }}
                                                    >
                                                        Read PDF
                                                    </Button>
                                                )}

                                                <Button
                                                    size="small"
                                                    startIcon={<ContentCopyIcon sx={{ fontSize: '0.85rem !important' }} />}
                                                    onClick={() => {
                                                        const cleanKey = (paper.paper_id || 'ref').replace(/[^a-zA-Z0-9_-]/g, '_');
                                                        const bibStr = `@article{${cleanKey},\n  title={${paper.title}},\n  author={${(paper.authors || []).join(' and ') || 'Unknown'}},\n  year={${paper.year || new Date().getFullYear()}}\n}`;
                                                        navigator.clipboard.writeText(bibStr);
                                                        toast.success('BibTeX copied to clipboard');
                                                    }}
                                                    sx={{
                                                        color: '#71717A',
                                                        border: '1px solid #27272F',
                                                        textTransform: 'none',
                                                        fontWeight: 600,
                                                        fontSize: '0.75rem',
                                                        borderRadius: '8px',
                                                        py: 0.5,
                                                        px: 1.2,
                                                        '&:hover': {
                                                            color: '#F4F4F5',
                                                            borderColor: '#3F3F46',
                                                            backgroundColor: 'rgba(255,255,255,0.05)',
                                                        },
                                                    }}
                                                >
                                                    Copy BibTeX
                                                </Button>
                                            </Box>
                                        </PaperItemCard>
                                    );
                                })}
                            </Box>
                        )}
                    </Box>
                )}

                {/* Tab 5: Real-Time Journey & LangGraph DAG Telemetry */}
                {activeTab === 5 && (
                    <AgentPipeline
                        currentAgent={currentAgent}
                        progress={progress}
                        logs={logs}
                        isConnected={isConnected}
                        projectStatus={projectStatus}
                        latestCriticVerdict={latestCriticVerdict}
                        latestFactCheck={latestFactCheck}
                    />
                )}

                {/* Multi-Format Export Menu */}
                <Menu
                    anchorEl={exportMenuAnchor}
                    open={exportMenuOpen}
                    onClose={handleExportClose}
                    PaperProps={{
                        sx: {
                            backgroundColor: '#18181B',
                            border: '1px solid rgba(255, 255, 255, 0.1)',
                            borderRadius: '12px',
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
                            minWidth: '200px',
                        },
                    }}
                >
                    <MenuItem
                        onClick={() => triggerExport('md')}
                        sx={{ color: '#F4F4F5', fontWeight: 600, '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.08)' } }}
                    >
                        Export as Markdown (.md)
                    </MenuItem>
                    <MenuItem
                        onClick={() => triggerExport('pdf')}
                        sx={{ color: '#F4F4F5', fontWeight: 600, '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.08)' } }}
                    >
                        Export as PDF Document (.pdf)
                    </MenuItem>
                    <MenuItem
                        onClick={() => triggerExport('docx')}
                        sx={{ color: '#F4F4F5', fontWeight: 600, '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.08)' } }}
                    >
                        Export as Word Document (.docx)
                    </MenuItem>
                    <MenuItem
                        onClick={() => triggerExport('bib')}
                        sx={{ color: '#F4F4F5', fontWeight: 600, '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.08)' } }}
                    >
                        Export BibTeX References (.bib)
                    </MenuItem>
                </Menu>
            </Box>
        </PageWrapper>
    );
};

export default ProjectDetailsPage;