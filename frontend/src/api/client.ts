import axios from 'axios';
import type { ResearchProject, ProjectCreate } from '../types';
import { neonClient } from './neonClient';

const apiClient = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
});

// Interceptor to add the auth token to every request
apiClient.interceptors.request.use(
    async (config) => {
        const { data } = await neonClient.auth.getSession();
        if (data?.session?.access_token) {
            config.headers.Authorization = `Bearer ${data.session.access_token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// --- Backend-dependent operations (require Render backend) ---

export const createProject = async (project: ProjectCreate): Promise<ResearchProject> => {
    const { data } = await apiClient.post('/api/projects', project);
    return data;
};

// Agent Execution
export const startLiteratureReview = async (projectId: string, maxPapers: number = 50): Promise<{ job_id: string, status: string }> => {
    const { data } = await apiClient.post(`/api/projects/${projectId}/start?max_papers=${maxPapers}`);
    return data;
};

// Health Check to verify API is running
export const checkHealth = async () => {
    try {
        const { data } = await apiClient.get('/api/health');
        return data.status === 'ok';
    } catch (error) {
        console.error("API health check failed:", error);
        return false;
    }
};