import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { User } from '../types';
import {type LoginCredentials, type RegisterCredentials } from '../api/client';
import { neonAuth, neonData } from '../api/neonClient';
import { useProjectStore } from './projectStore';
import { toast } from 'react-toastify';

interface AuthState {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
    isInitialized: boolean;
    fetchUser: () => Promise<void>; 
    login: (credentials: LoginCredentials) => Promise<boolean>;
    loginWithOAuth: (provider: 'google' | 'github') => Promise<void>;
    register: (credentials: RegisterCredentials) => Promise<boolean>;
    logout: () => void;
    setToken: (token: string | null) => void;
    initialize: () => Promise<void>;
}

export const useAuthStore = create(
    persist<AuthState>(
        (set, get) => ({
            user: null,
            token: null,
            isAuthenticated: false,
            isInitialized: false,
            initialize: async () => {
                try {
                    const { data: sessionData } = await neonAuth.getSession();
                    if (sessionData?.session) {
                        const token = sessionData.session.access_token;
                        const profile = await neonData.getProfile();
                        set({ 
                            user: profile, 
                            token,
                            isAuthenticated: true,
                            isInitialized: true,
                        });
                    } else {
                        set({ 
                            user: null, 
                            token: null, 
                            isAuthenticated: false,
                            isInitialized: true,
                        });
                    }
                } catch (error) {
                    console.error("Failed to initialize auth:", error);
                    set({ 
                        user: null, 
                        token: null, 
                        isAuthenticated: false,
                        isInitialized: true,
                    });
                }
            },
            fetchUser: async () => {
                try {
                    const { data: sessionData } = await neonAuth.getSession();
                    if (sessionData?.session?.user) {
                        const token = sessionData.session.access_token;
                        const profile = await neonData.getProfile();
                        set({ user: profile, token, isAuthenticated: true });
                    } else {
                        set({ user: null, token: null, isAuthenticated: false });
                    }
                } catch (error) {
                    console.error("Failed to fetch user, token might be invalid:", error);
                    set({ user: null, token: null, isAuthenticated: false });
                }
            },
            setToken: (token) => {
                set({
                    token,
                    isAuthenticated: !!token,
                });
            },
            login: async (credentials) => {
                try {
                    const { data, error } = await neonAuth.signIn(credentials.username, credentials.password);
                    if (error) throw error;
                    const token = data?.session?.access_token || null;
                    set({ token, isAuthenticated: !!token });
                    try {
                        useProjectStore.getState().setProjects([]);
                    } catch {
                        // ignore if project store isn't available
                    }
                    await get().fetchUser();
                    return true;
                } catch (error: unknown) {
                    const errorMessage = (error as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail 
                        || (error as { message?: string })?.message 
                        || "Login failed. Please check your credentials.";
                    toast.error(errorMessage);
                    return false;
                }
            },
            loginWithOAuth: async (provider) => {
                try {
                    const { error } = await neonAuth.signInWithOAuth(provider);
                    if (error) throw error;
                } catch (error: unknown) {
                    const errorMessage = (error as { message?: string })?.message 
                        || `OAuth login with ${provider} failed.`;
                    toast.error(errorMessage);
                }
            },
            register: async (credentials) => {
                try {
                    const { error } = await neonAuth.signUp(credentials.email, credentials.password, credentials.name);
                    if (error) throw error;
                    return await get().login({ username: credentials.email, password: credentials.password });
                } catch (error: unknown) {
                    const errorMessage = (error as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail 
                        || (error as { message?: string })?.message 
                        || "Registration failed.";
                    toast.error(errorMessage);
                    return false;
                }
            },
            logout: () => {
                neonAuth.signOut().catch(console.error);
                set({ user: null, token: null, isAuthenticated: false });
                try {
                    useProjectStore.getState().setProjects([]);
                } catch {
                    // ignore
                }
                toast.info("You have been logged out.");
            },
        }),
        {
            name: 'auth-storage',
            storage: createJSONStorage(() => localStorage),
            partialize: (state) => ({ 
                token: state.token, 
                isAuthenticated: state.isAuthenticated 
            } as AuthState),
            onRehydrateStorage: () => (state) => {
                if (state) {
                    state.initialize();
                }
            },
        }
    )
);
