import { createClient, SupabaseAuthAdapter } from '@neondatabase/neon-js';
import { NEON_AUTH_URL, NEON_DATA_API_URL, RENDER_BACKEND_URL } from '../config';
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

async function syncUserWithBackend(): Promise<void> {
  const { data } = await neonClient.auth.getSession();
  if (!data?.session?.access_token) return;
  
  try {
    const response = await fetch(`${RENDER_BACKEND_URL}/api/auth/users/me`, {
      headers: {
        'Authorization': `Bearer ${data.session.access_token}`,
      },
    });
    if (!response.ok && response.status !== 401) {
      console.warn('Backend user sync returned status:', response.status);
    }
  } catch (error) {
    console.warn('Backend user sync failed (backend may be waking up):', error);
  }
}

export const neonData = {
  getProfile: async (): Promise<User | null> => {
    const { data: userData } = await neonClient.auth.getUser();
    if (!userData?.user) return null;
    
    await syncUserWithBackend();
    
    const neonUser = userData.user;
    return {
      id: neonUser.id,
      email: neonUser.email || '',
      name: neonUser.user_metadata?.name || neonUser.email?.split('@')[0] || 'User',
    };
  },

  getProjects: async (): Promise<ResearchProject[]> => {
    const { data } = await neonClient.auth.getSession();
    const token = data?.session?.access_token;

    try {
      const res = await fetch(`${RENDER_BACKEND_URL}/api/projects`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });

      if (!res.ok) {
        throw new Error(`Failed to fetch projects: ${res.status}`);
      }

      const projects = await res.json();
      return projects as ResearchProject[];
    } catch (error) {
      console.warn('Failed to fetch projects (backend may be waking up):', error);
      throw error;
    }
  },
};
