import { useEffect, useRef } from 'react';
import { neonData } from '../api/neonClient';
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
                    const updatedProject = await neonData.getProjectById(projectId);
                    const currentProject = useProjectStore.getState().projects.find(p => p.id === projectId);

                    if (updatedProject && JSON.stringify(currentProject) !== JSON.stringify(updatedProject)) {
                        updateProject(updatedProject); // Update the entire project object

                        // --- REMOVED THE NOISY INTERMEDIATE STATUS TOAST ---
                        // The ProgressTracker component already shows the current status.
                        // We only notify on terminal states (completed or error).

                        if (!['searching', 'analyzing', 'synthesizing', 'planning'].includes(updatedProject.status)) {
                            if (updatedProject.status === 'completed') {
                                // Add a toastId to prevent duplicate notifications
                                toast.success(`Project "${updatedProject.title}" has completed. A report has been sent to your email address.`, {
                                    toastId: `${updatedProject.id}-completed`
                                });
                            } else if (updatedProject.status.startsWith('error')) {
                                // Add a toastId to prevent duplicate notifications
                                toast.error(`Project "${updatedProject.title}" encountered an error.`, {
                                    toastId: `${updatedProject.id}-error`
                                });
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