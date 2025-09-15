import axios from 'axios';
import type { ResearchProject, ProjectCreate, User } from '../types';
import { useAuthStore } from '../store/authStore';

const apiClient = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
});

// Interceptor to add the auth token to every request
apiClient.interceptors.request.use(
    (config) => {
        const token = useAuthStore.getState().token;
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// --- New Auth Types and Functions ---
export interface LoginCredentials {
    username: string; // FastAPI's OAuth2 expects 'username'
    password: string;
}

export interface RegisterCredentials {
    email: string;
    password: string;
    name: string;
}

export const login = async (credentials: LoginCredentials): Promise<{ access_token: string, token_type: string }> => {
    const params = new URLSearchParams();
    params.append('username', credentials.username);
    params.append('password', credentials.password);
    
    const { data } = await apiClient.post('/api/auth/token', params, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    return data;
};

export const register = async (credentials: RegisterCredentials): Promise<User> => {
    const { data } = await apiClient.post('/api/auth/register', credentials);
    return data;
};

export const getMe = async (): Promise<User> => {
    const { data } = await apiClient.get('/api/auth/users/me');
    return data;
};


// --- Project Management ---
export const createProject = async (project: ProjectCreate): Promise<ResearchProject> => {
    const { data } = await apiClient.post('/api/projects', project);
    return data;
};

// ... rest of the file remains the same ...
export const getProjects = async (): Promise<ResearchProject[]> => {
    // Fetches all projects from the backend.
    const { data } = await apiClient.get('/api/projects');
    return data;
};


export const getProjectById = async (projectId: string): Promise<ResearchProject> => {
    // Fetches a single project by its unique ID.
    const { data } = await apiClient.get(`/api/projects/${projectId}`);
    return data;
}


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