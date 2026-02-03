import { create } from 'zustand';
import type { ResearchProject, ProjectCreate } from '../types';
import { createProject as apiCreateProject, getProjects, deleteProject as apiDeleteProject } from '../api/client';
import { toast } from 'react-toastify';

interface ProjectState {
    projects: ResearchProject[];
    isLoading: boolean;
    error: string | null;
    fetchProjects: () => Promise<void>;
    addProject: (newProjectData: ProjectCreate) => Promise<void>;
    removeProject: (projectId: string) => Promise<boolean>;
    setProjects: (projects: ResearchProject[]) => void;
    updateProject: (updatedProject: ResearchProject) => void;
    updateProjectStatus: (projectId: string, status: ResearchProject['status']) => void;
}

export const useProjectStore = create<ProjectState>((set, _get) => ({
    projects: [],
    isLoading: false,
    error: null,
    fetchProjects: async () => {
        set({ isLoading: true });
        try {
            const projects = await getProjects();
            set({ projects, isLoading: false });
        } catch {
            const errorMessage = 'Failed to fetch projects.';
            set({ error: errorMessage, isLoading: false });
            toast.error(errorMessage);
        }
    },
    addProject: async (newProjectData) => {
        // Create an optimistic project to show in the UI immediately.
        const optimisticProject: ResearchProject = {
            id: `temp-${Date.now()}`,
            title: newProjectData.title,
            research_question: newProjectData.research_question,
            status: 'creating',
            keywords: [],
            agent_plans: [],
            paper_references: [],
            created_at: '',
            subtopics: [],
            total_papers_found: 0
        };

        // Add the optimistic project to the state right away.
        set((state) => ({
            projects: [optimisticProject, ...state.projects],
        }));

        try {
            // In the background, create the project on the server.
            const newProject = await apiCreateProject(newProjectData);
            
            // Once successful, replace the optimistic project with the real one.
            set((state) => ({
                projects: state.projects.map((p) =>
                    p.id === optimisticProject.id ? newProject : p
                ),
            }));
            toast.success(`Project "${newProject.title}" created successfully!`);
        } catch (error) {
            console.error("Failed to create project:", error);
            const errorMessage = "Failed to create project. Please check the connection to the backend.";
            
            // If it fails, remove the optimistic project from the list.
            set((state) => ({
                projects: state.projects.filter((p) => p.id !== optimisticProject.id),
                error: errorMessage
            }));
            toast.error(errorMessage);
        }
    },
    removeProject: async (projectId) => {
        // Store project info for potential rollback
        const state = _get();
        const projectToDelete = state.projects.find((p) => p.id === projectId);
        
        if (!projectToDelete) {
            toast.error("Project not found");
            return false;
        }

        // Optimistically remove from UI
        set((state) => ({
            projects: state.projects.filter((p) => p.id !== projectId),
        }));

        try {
            await apiDeleteProject(projectId);
            toast.success(`Project "${projectToDelete.title}" deleted successfully`);
            return true;
        } catch (error) {
            console.error("Failed to delete project:", error);
            // Rollback: restore the project
            set((state) => ({
                projects: [...state.projects, projectToDelete],
            }));
            toast.error("Failed to delete project. Please try again.");
            return false;
        }
    },
    updateProjectStatus: (projectId, status) => {
        set((state) => ({
            projects: state.projects.map((p) =>
                p.id === projectId ? { ...p, status } : p
            ),
        }));
    },
    updateProject: (updatedProject) => {
        set((state) => ({
            projects: state.projects.map((p) =>
                p.id === updatedProject.id ? updatedProject : p
            ),
        }));
    },
    setProjects: (projects) => set({ projects }),
}));