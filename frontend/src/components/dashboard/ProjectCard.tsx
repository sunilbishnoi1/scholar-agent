import React from 'react';
import { Card, CardContent, Typography, Button, CardActions, Box, LinearProgress } from '@mui/material';
import { Link } from 'react-router-dom';
import type { ResearchProject } from '../../types';
import StatusChip from '../common/StatusChip';
import { startLiteratureReview } from '../../api/client';
import { useProjectStore } from '../../store/projectStore';
import { toast } from 'react-toastify';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useProjectStatusPoller } from '../../hooks/useProjectStatusPoller';
    import { useNavigate } from 'react-router-dom';

interface ProjectCardProps {
    project: ResearchProject;
}

const ProgressTracker: React.FC<{ project: ResearchProject }> = ({ project }) => {
    const papersAnalyzed = project.agent_plans.filter(p => p.agent_type === 'analyzer').length;
    const totalPapersToAnalyze = project.total_papers_found;

    let progress = 0;
    let progressText = 'Initializing...';
    let isIndeterminate = false;

    const BASE_SEARCHING = 5;
    const BASE_ANALYZING = 15;
    const BASE_SYNTHESIZING = 95;

    switch (project.status) {
        case 'planning':
            progress = 2;
            progressText = 'Initializing workflow...';
            isIndeterminate = true;
            break;
        case 'searching':
            progress = BASE_SEARCHING;
            progressText = `Searching for relevant papers...`;
            isIndeterminate = true;
            break;
        case 'analyzing':
            if (totalPapersToAnalyze > 0) {
                progress = BASE_ANALYZING + (papersAnalyzed / totalPapersToAnalyze) * (BASE_SYNTHESIZING - BASE_ANALYZING);
                progressText = `Analyzing: ${papersAnalyzed} of ${totalPapersToAnalyze} papers`;
            } else {
                progress = BASE_ANALYZING;
                progressText = "Preparing to analyze papers...";
                isIndeterminate = true;
            }
            break;
        case 'synthesizing':
            progress = BASE_SYNTHESIZING;
            progressText = 'Synthesizing final report...';
            isIndeterminate = true;
            break;
        default:
            return null;
    }

    return (
        <Box className="w-full px-2">
            <Typography variant="caption" display="block" gutterBottom align="left" className="text-slate-600">
                {progressText}
            </Typography>
            <LinearProgress
                variant={isIndeterminate ? 'indeterminate' : 'determinate'}
                value={progress}
                className="h-2 rounded-full"
            />
        </Box>
    );
};


const ProjectCard: React.FC<ProjectCardProps> = ({ project }) => {
    const { updateProjectStatus } = useProjectStore();
    const isProcessing = ['planning', 'searching', 'analyzing', 'synthesizing'].includes(project.status);
    useProjectStatusPoller(isProcessing ? project.id : undefined);

    const handleStartReview = async () => {
        try {
            toast.info(`Starting literature review for "${project.title}"...`);
            updateProjectStatus(project.id, 'planning');
            await startLiteratureReview(project.id);
        } catch (error) {
            console.error('Failed to start literature review:', error);
            toast.error('Failed to start literature review. Please try again.');
            updateProjectStatus(project.id, 'error');
        }
    };


    const navigate = useNavigate();
    const handleCardClick = async () => {
    if (project.status === 'completed') {
        navigate(`/project/${project.id}`);
    }
    };

    const isReady = project.status === 'created';
    const isFailed = project.status === 'error' || project.status === 'error_no_papers_found';

    return (
        <Card className="flex flex-col h-full bg-white/60 backdrop-blur-lg border border-gray-200/50" onClick={handleCardClick}>
            <CardContent className="flex-grow">
                <Box className="flex justify-between items-start mb-2">
                    <Typography variant="h6" component="div" className="font-bold text-slate-800 text-left">
                        {project.title}
                    </Typography>
                    <StatusChip status={project.status} />
                </Box>
                <Typography variant="body2" color="text.secondary" className="italic mb-2 text-left">
                    "{project.research_question}"
                </Typography>
            </CardContent>
            <CardActions className="bg-slate-50/50 p-4 flex justify-between items-center min-h-[80px]">
                {isReady && (
                    <Button
                        size="small"
                        variant="contained"
                        onClick={handleStartReview}
                        startIcon={<AutoAwesomeIcon />}
                        className='bg-gradient-to-r from-blue-600 to-teal-500 hover:bg-blue-700 text-white'
                    >
                        Start Review
                    </Button>
                )}
                {isProcessing && <ProgressTracker project={project} />}
                {project.status === 'completed' && (
                    <Button
                        component={Link}
                        to={`/project/${project.id}`}
                        variant="contained"
                        size="small"
                        className="bg-gradient-to-r from-blue-600 to-blue-500 hover:bg-blue-700 text-white"
                        onClick={(e) => e.stopPropagation()}
                        >
                        View Results
                        </Button>
                )}
                 {isFailed && (
                     <Button
                        size="small"
                        variant="contained"
                        color="secondary"
                        onClick={handleStartReview}
                        startIcon={<AutoAwesomeIcon />}
                    >
                        Retry Review
                    </Button>
                 )}
            </CardActions>
        </Card>
    );
};

export default ProjectCard;