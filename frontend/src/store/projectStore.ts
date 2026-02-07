import { create } from 'zustand';
import type { ResearchProject, ProjectCreate } from '../types';
import { createProject, deleteProject } from '../api/client';
import { neonData } from '../api/neonClient';
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
            const projects = await neonData.getProjects();
            set({ projects, isLoading: false });
        } catch {
            const errorMessage = 'Failed to fetch projects. Neon database might be unavailable.';
            set({ error: errorMessage, isLoading: false });
            toast.error(errorMessage);
        }
    },
    addProject: async (newProjectData) => {
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

        set((state) => ({
            projects: [optimisticProject, ...state.projects],
        }));

        try {
            const newProject = await createProject(newProjectData);
            set((state) => ({
                projects: state.projects.map((p) =>
                    p.id === optimisticProject.id ? newProject : p
                ),
            }));
            toast.success(`Project "${newProject.title}" created successfully!`);
        } catch (error) {
            console.error("Failed to create project:", error);
            const errorMessage = "Failed to create project. Backend may be starting up.";
            set((state) => ({
                projects: state.projects.filter((p) => p.id !== optimisticProject.id),
                error: errorMessage
            }));
            toast.error(errorMessage);
        }
    },
    removeProject: async (projectId) => {
        const state = _get();
        const projectToDelete = state.projects.find((p) => p.id === projectId);

        if (!projectToDelete) {
            toast.error("Project not found");
            return false;
        }

        set((state) => ({
            projects: state.projects.filter((p) => p.id !== projectId),
        }));

        try {
            await deleteProject(projectId);
            toast.success(`Project "${projectToDelete.title}" deleted successfully`);
            return true;
        } catch (error) {
            console.error("Failed to delete project:", error);
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