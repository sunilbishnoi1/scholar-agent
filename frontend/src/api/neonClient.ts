import { createClient, SupabaseAuthAdapter } from '@neondatabase/neon-js';
import { NEON_AUTH_URL, NEON_DATA_API_URL } from '../config';
import type { User, ResearchProject } from '../types';

const getOAuthCallbackURL = () => {
  if (typeof window !== 'undefined') {
    return `${window.location.origin}/auth/callback`;
  }
  return 'http://localhost:5173/auth/callback';
};

export const neonClient = createClient({
  auth: {
    adapter: SupabaseAuthAdapter(),
    url: NEON_AUTH_URL,
  },
  dataApi: {
    url: NEON_DATA_API_URL,
  },
});

export const neonAuth = {
  signUp: async (email: string, password: string, name: string) => {
    const result = await neonClient.auth.signUp({
      email,
      password,
      options: {
        data: { name },
      },
    });
    return result;
  },

  signIn: async (email: string, password: string) => {
    const result = await neonClient.auth.signInWithPassword({
      email,
      password,
    });
    return result;
  },

  signOut: async () => {
    return neonClient.auth.signOut();
  },

  getSession: async () => {
    return neonClient.auth.getSession();
  },

  getUser: async () => {
    return neonClient.auth.getUser();
  },

  onAuthStateChange: (callback: (event: string, session: unknown) => void) => {
    return neonClient.auth.onAuthStateChange(callback);
  },

  signInWithOAuth: async (provider: 'google' | 'github') => {
    const result = await neonClient.auth.signInWithOAuth({
      provider,
      options: {
        redirectTo: getOAuthCallbackURL(),
      },
    });
    return result;
  },
};

export const neonData = {
  getProfile: async (): Promise<User | null> => {
    const { data: userData } = await neonClient.auth.getUser();
    if (!userData?.user) return null;

    const neonUser = userData.user;
    return {
      id: neonUser.id,
      email: neonUser.email || '',
      name: neonUser.user_metadata?.name || neonUser.email?.split('@')[0] || 'User',
    };
  },

  getProjects: async (): Promise<ResearchProject[]> => {
    const { data, error } = await neonClient
      .from('research_projects')
      .select('*, agent_plans(*), paper_references(*)')
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Failed to fetch projects from Neon:', error);
      throw new Error(`Failed to fetch projects: ${error.message}`);
    }

    return (data || []) as unknown as ResearchProject[];
  },

  getProjectById: async (projectId: string): Promise<ResearchProject> => {
    const { data, error } = await neonClient
      .from('research_projects')
      .select('*, agent_plans(*), paper_references(*)')
      .eq('id', projectId)
      .single();

    if (error) {
      console.error('Failed to fetch project from Neon:', error);
      throw new Error(`Failed to fetch project: ${error.message}`);
    }

    return data as unknown as ResearchProject;
  },

  deleteProject: async (projectId: string): Promise<{ id: string; deleted: boolean; message: string }> => {
    // Delete associated records first (PostgREST doesn't support transactions,
    // but cascading deletes should be handled by FK constraints in the DB.
    // We delete explicitly for safety in case ON DELETE CASCADE isn't set.)

    const { error: llmError } = await neonClient
      .from('llm_interactions')
      .delete()
      .eq('project_id', projectId);

    if (llmError) {
      console.warn('Failed to delete LLM interactions:', llmError);
    }

    const { error: plansError } = await neonClient
      .from('agent_plans')
      .delete()
      .eq('project_id', projectId);

    if (plansError) {
      console.warn('Failed to delete agent plans:', plansError);
    }

    const { error: papersError } = await neonClient
      .from('paper_references')
      .delete()
      .eq('project_id', projectId);

    if (papersError) {
      console.warn('Failed to delete paper references:', papersError);
    }

    const { error: projectError } = await neonClient
      .from('research_projects')
      .delete()
      .eq('id', projectId);

    if (projectError) {
      console.error('Failed to delete project from Neon:', projectError);
      throw new Error(`Failed to delete project: ${projectError.message}`);
    }

    return {
      id: projectId,
      deleted: true,
      message: 'Project and associated data deleted successfully',
    };
  },
};
