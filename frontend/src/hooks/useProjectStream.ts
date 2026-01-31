import { useEffect, useState, useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

// Event types from backend
export type EventType =
    | 'connected'
    | 'disconnected'
    | 'agent_started'
    | 'agent_completed'
    | 'agent_error'
    | 'status'
    | 'progress'
    | 'log'
    | 'paper_found'
    | 'paper_analyzed'
    | 'complete'
    | 'error'
    | 'pong';

export interface AgentUpdate {
    type: EventType;
    agent?: string;
    project_id?: string;
    message?: string;
    progress?: number;
    data?: Record<string, unknown>;
    timestamp?: string;
}

export interface UseProjectStreamOptions {
    /** Token for WebSocket authentication */
    token?: string;
    /** Whether to automatically reconnect on disconnect */
    autoReconnect?: boolean;
    /** Reconnect delay in milliseconds */
    reconnectDelay?: number;
    /** Maximum reconnect attempts */
    maxReconnectAttempts?: number;
}

export interface UseProjectStreamReturn {
    /** Whether WebSocket is connected */
    isConnected: boolean;
    /** All received updates */
    updates: AgentUpdate[];
    /** Currently active agent */
    currentAgent: string | null;
    /** Overall progress percentage (0-100) */
    progress: number;
    /** Latest log messages */
    logs: string[];
    /** Manually connect to WebSocket */
    connect: () => void;
    /** Manually disconnect from WebSocket */
    disconnect: () => void;
    /** Clear all updates */
    clearUpdates: () => void;
}

/**
 * Hook for real-time project updates via WebSocket.
 * 
 * Replaces polling with streaming updates for better UX.
 * 
 * @example
 * ```tsx
 * const { isConnected, currentAgent, progress, logs } = useProjectStream(projectId, {
 *   token: authToken,
 *   autoReconnect: true,
 * });
 * ```
 */
export function useProjectStream(
    projectId: string | undefined,
    options: UseProjectStreamOptions = {}
): UseProjectStreamReturn {
    const {
        token,
        autoReconnect = true,
        reconnectDelay = 3000,
        maxReconnectAttempts = 5,
    } = options;

    const [isConnected, setIsConnected] = useState(false);
    const [updates, setUpdates] = useState<AgentUpdate[]>([]);
    const [currentAgent, setCurrentAgent] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);
    const [logs, setLogs] = useState<string[]>([]);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectAttemptsRef = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const queryClient = useQueryClient();

    const clearUpdates = useCallback(() => {
        setUpdates([]);
        setLogs([]);
        setProgress(0);
        setCurrentAgent(null);
    }, []);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    const connect = useCallback(() => {
        if (!projectId) return;
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        // Build WebSocket URL
        const baseUrl = import.meta.env.VITE_WS_URL || 
            (import.meta.env.VITE_API_BASE_URL?.replace('http', 'ws') || 'ws://localhost:8000');
        
        let wsUrl = `${baseUrl}/ws/projects/${projectId}/stream`;
        if (token) {
            wsUrl += `?token=${encodeURIComponent(token)}`;
        }

        console.log('[WebSocket] Connecting to:', wsUrl);

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('[WebSocket] Connected');
            setIsConnected(true);
            reconnectAttemptsRef.current = 0;

            // Set up ping interval to keep connection alive
            pingIntervalRef.current = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send('ping');
                }
            }, 30000);
        };

        ws.onmessage = (event) => {
            try {
                const update: AgentUpdate = JSON.parse(event.data);
                console.log('[WebSocket] Received:', update);

                // Skip pong messages
                if (update.type === 'pong') return;

                setUpdates((prev) => [...prev.slice(-99), update]); // Keep last 100 updates

                switch (update.type) {
                    case 'agent_started':
                    case 'status':
                        if (update.agent) {
                            setCurrentAgent(update.agent);
                        }
                        break;

                    case 'agent_completed':
                        // Keep current agent visible until next one starts
                        break;

                    case 'progress':
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'log':
                    case 'paper_found':
                    case 'paper_analyzed':
                        if (update.message) {
                            setLogs((prev) => [...prev.slice(-49), update.message!]);
                        }
                        // Also update progress if provided
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'complete':
                        setProgress(100);
                        setCurrentAgent(null);
                        // Invalidate project query to refresh data
                        queryClient.invalidateQueries({ queryKey: ['project', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['projects'] });
                        break;

                    case 'error':
                        setCurrentAgent(null);
                        // Invalidate project query to show error state
                        queryClient.invalidateQueries({ queryKey: ['project', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['projects'] });
                        break;
                }
            } catch (e) {
                console.error('[WebSocket] Failed to parse message:', e);
            }
        };

        ws.onclose = (event) => {
            console.log('[WebSocket] Closed:', event.code, event.reason);
            setIsConnected(false);
            wsRef.current = null;

            if (pingIntervalRef.current) {
                clearInterval(pingIntervalRef.current);
                pingIntervalRef.current = null;
            }

            // Auto-reconnect logic
            if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
                reconnectAttemptsRef.current++;
                console.log(`[WebSocket] Reconnecting in ${reconnectDelay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);
                reconnectTimeoutRef.current = setTimeout(connect, reconnectDelay);
            }
        };

        ws.onerror = (error) => {
            console.error('[WebSocket] Error:', error);
        };
    }, [projectId, token, autoReconnect, reconnectDelay, maxReconnectAttempts, queryClient]);

    // Connect when projectId changes
    useEffect(() => {
        if (projectId) {
            connect();
        }
        return () => {
            disconnect();
        };
    }, [projectId, connect, disconnect]);

    return {
        isConnected,
        updates,
        currentAgent,
        progress,
        logs,
        connect,
        disconnect,
        clearUpdates,
    };
}

export default useProjectStream;
