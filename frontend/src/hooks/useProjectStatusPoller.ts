import { useEffect, useRef } from 'react';
import { getProjectById } from '../api/client';
import { useProjectStore } from '../store/projectStore';
import { toast } from 'react-toastify';

const POLLING_INTERVAL = 5000; // Poll every 5 seconds

/**
 * A hook to poll the status of a given project ID as long as it's in a running state.
 * @param projectId The ID of the project to poll.
 */
export const useProjectStatusPoller = (projectId?: string) => {
    const { updateProject, projects } = useProjectStore();
    const pollingInterval = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        const project = projects.find((p) => p.id === projectId);
        const isRunning = project && ['searching', 'analyzing', 'synthesizing', 'planning'].includes(project.status);

        const stopPolling = () => {
            if (pollingInterval.current) {
                clearInterval(pollingInterval.current);
                pollingInterval.current = null;
            }
        };

        if (projectId && isRunning) {
            if (pollingInterval.current) return;

            pollingInterval.current = setInterval(async () => {
                try {
                    const updatedProject = await getProjectById(projectId);
                    const currentProject = useProjectStore.getState().projects.find(p => p.id === projectId);

                    if (updatedProject && JSON.stringify(currentProject) !== JSON.stringify(updatedProject)) {
                        updateProject(updatedProject); // Update the entire project object

                        if (currentProject?.status !== updatedProject.status) {
                             toast.info(`Project "${updatedProject.title}" status: ${updatedProject.status}`);
                        }

                        if (!['searching', 'analyzing', 'synthesizing', 'planning'].includes(updatedProject.status)) {
                            if (updatedProject.status === 'completed') {
                                toast.success(`Project "${updatedProject.title}" has completed!`);
                            } else if (updatedProject.status.startsWith('error')) {
                                toast.error(`Project "${updatedProject.title}" encountered an error.`);
                            }
                            stopPolling();
                        }
                    }
                } catch (error) {
                    console.error('Polling error:', error);
                    toast.error('Error polling for project status.');
                    stopPolling();
                }
            }, POLLING_INTERVAL);
        } else {
            stopPolling();
        }

        return () => {
            stopPolling();
        };
    }, [projectId, projects, updateProject]);
};