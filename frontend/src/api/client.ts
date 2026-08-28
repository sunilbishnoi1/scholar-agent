import axios from 'axios';
import type {
    ResearchProject,
    ProjectCreate,
    ReportResponse,
    MatrixResponse,
    GapsResponse,
    PaperSectionsResponse,
} from '../types';
import { neonClient } from './neonClient';

const apiClient = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
});

// Interceptor to add the auth token to every request
apiClient.interceptors.request.use(
    async (config) => {
        try {
            const { data } = await neonClient.auth.getSession();
            if (data?.session?.access_token) {
                config.headers.Authorization = `Bearer ${data.session.access_token}`;
            }
        } catch {
            // Ignore auth session retrieval failure in local/test modes
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// --- Backend REST API Operations (Scholar Agent v3.2) ---

export const getProjects = async (): Promise<ResearchProject[]> => {
    const { data } = await apiClient.get('/api/projects');
    return data;
};

export const getProject = async (projectId: string): Promise<ResearchProject> => {
    const { data } = await apiClient.get(`/api/projects/${projectId}`);
    return data;
};

export const createProject = async (project: ProjectCreate): Promise<ResearchProject> => {
    const { data } = await apiClient.post('/api/projects', project);
    return data;
};

export const deleteProject = async (projectId: string): Promise<{ id: string; deleted: boolean; message: string }> => {
    const { data } = await apiClient.delete(`/api/projects/${projectId}`);
    return data;
};

// Agent Execution
export const startLiteratureReview = async (
    projectId: string,
    maxPapers?: number
): Promise<{ job_id: string; status: string; estimated_duration?: string }> => {
    const query = maxPapers ? `?max_papers=${maxPapers}` : '';
    const { data } = await apiClient.post(`/api/projects/${projectId}/start${query}`);
    return data;
};

export const stopLiteratureReview = async (
    projectId: string
): Promise<{ status: string; project_id: string; message: string }> => {
    const { data } = await apiClient.post(`/api/projects/${projectId}/stop`);
    return data;
};

// Research Report Deliverables
export const getProjectReport = async (projectId: string): Promise<ReportResponse> => {
    const { data } = await apiClient.get(`/api/projects/${projectId}/report`);
    return data;
};

export const getProjectMatrix = async (projectId: string): Promise<MatrixResponse> => {
    const { data } = await apiClient.get(`/api/projects/${projectId}/matrix`);
    return data;
};

export const getProjectGaps = async (projectId: string): Promise<GapsResponse> => {
    const { data } = await apiClient.get(`/api/projects/${projectId}/gaps`);
    return data;
};

export const getPaperSections = async (paperId: string): Promise<PaperSectionsResponse> => {
    const { data } = await apiClient.get(`/api/papers/${encodeURIComponent(paperId)}/sections`);
    return data;
};

// Health Check to verify API is running
export const checkHealth = async (): Promise<boolean> => {
    try {
        const { data } = await apiClient.get('/api/health');
        return data.status === 'ok';
    } catch (error) {
        console.error('API health check failed:', error);
        return false;
    }
};

export default apiClient;