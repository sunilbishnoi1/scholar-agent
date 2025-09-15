import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { User } from '../types';
import { login as apiLogin, register as apiRegister, getMe, type LoginCredentials, type RegisterCredentials } from '../api/client';
import { toast } from 'react-toastify';

interface AuthState {
    user: User | null;
    token: string | null;
    isAuthenticated: boolean;
    fetchUser: () => Promise<void>; 
    login: (credentials: LoginCredentials) => Promise<boolean>;
    register: (credentials: RegisterCredentials) => Promise<boolean>;
    logout: () => void;
    setToken: (token: string | null) => void;
}

export const useAuthStore = create(
    persist<AuthState>(
        (set, get) => ({
            user: null,
            token: null,
            isAuthenticated: false,
            // Fetches user data from the API using the stored token
            fetchUser: async () => {
                try {
                    const user = await getMe();
                    set({ user });
                } catch (error) {
                    console.error("Failed to fetch user, token might be invalid:", error);
                    // If the token is invalid, log the user out completely
                    set({ user: null, token: null, isAuthenticated: false });
                }
            },
            // This function is used to initialize or update the token and auth status
            setToken: (token) => {
                set({
                    token,
                    isAuthenticated: !!token,
                });
            },
            login: async (credentials) => {
                try {
                    const { access_token } = await apiLogin(credentials);
                    set({ token: access_token, isAuthenticated: true });
                    await get().fetchUser(); // Fetch user data immediately after a successful login
                    toast.success('Login successful!');
                    return true;
                } catch (error: any) {
                    const errorMessage = error.response?.data?.detail || "Login failed. Please check your credentials.";
                    toast.error(errorMessage);
                    return false;
                }
            },
            register: async (credentials) => {
                try {
                    await apiRegister(credentials);
                    // After successful registration, log the user in automatically.
                    const { email, password } = credentials;
                    // The login function handles setting the token, user state, and showing a success toast.
                    // LoginCredentials expects { username, password }, using email as username
                    return await get().login({ username: email, password });
                } catch (error: any) {
                    const errorMessage = error.response?.data?.detail || "Registration failed.";
                    toast.error(errorMessage);
                    return false;
                }
            },
            logout: () => {
                set({ user: null, token: null, isAuthenticated: false });
                toast.info("You have been logged out.");
            },
        }),
        {
            name: 'auth-storage', // The key in localStorage
            storage: createJSONStorage(() => localStorage),
            // Only persist the token. User data is fetched on load for freshness.
            
            partialize: (state) => ({ token: state.token, isAuthenticated: state.isAuthenticated } as AuthState), // persist token and authentication status

            // After the store is rehydrated from localStorage, this runs.
            onRehydrateStorage: () => (state) => {
                // This ensures the `isAuthenticated` flag is correctly set based on the persisted token.
                if (state) {
                    state.setToken(state.token);
                }
            },
        }
    )
);
