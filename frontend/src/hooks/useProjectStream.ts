import { useEffect, useState, useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useProjectStore } from '../store/projectStore';
import { useAuthStore } from '../store/authStore';
import type { ResearchProject, EventType, AgentWebSocketEvent } from '../types';

export type { EventType };
export type AgentUpdate = AgentWebSocketEvent;

export interface CriticVerdictData {
    score: number;
    should_refine: boolean;
    iteration: number;
    dimension_scores?: Record<string, number>;
    weaknesses?: string[];
    guidance?: string;
}

export interface FactCheckedData {
    precision_score: number;
    passed: boolean;
    entailed_count?: number;
    neutral_count?: number;
    contradiction_count?: number;
    total_propositions?: number;
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
    /** Number of papers analyzed/parsed (from real-time events) */
    papersAnalyzed: number;
    /** Total papers to analyze (from real-time events) */
    totalPapers: number;
    /** Latest critic verdict details if evaluated */
    latestCriticVerdict: CriticVerdictData | null;
    /** Latest fact check audit details if completed */
    latestFactCheck: FactCheckedData | null;
    /** Manually connect to WebSocket */
    connect: () => void;
    /** Manually disconnect from WebSocket */
    disconnect: () => void;
    /** Clear all updates */
    clearUpdates: () => void;
}

/**
 * Hook for real-time project updates via WebSocket (Scholar Agent v3.2).
 * 
 * Supports both standard v3.2 multi-agent DAG events and legacy events.
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
    const [papersAnalyzed, setPapersAnalyzed] = useState(0);
    const [totalPapers, setTotalPapers] = useState(0);
    const [latestCriticVerdict, setLatestCriticVerdict] = useState<CriticVerdictData | null>(null);
    const [latestFactCheck, setLatestFactCheck] = useState<FactCheckedData | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectAttemptsRef = useRef(0);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const queryClient = useQueryClient();
    const { updateProjectStatus } = useProjectStore();

    const clearUpdates = useCallback(() => {
        setUpdates([]);
        setLogs([]);
        setProgress(0);
        setCurrentAgent(null);
        setPapersAnalyzed(0);
        setTotalPapers(0);
        setLatestCriticVerdict(null);
        setLatestFactCheck(null);
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
        if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
            return;
        }

        // Build WebSocket URL
        const baseUrl = import.meta.env.VITE_WS_URL || 
            (import.meta.env.VITE_API_BASE_URL?.replace('http', 'ws') || 'ws://localhost:8000');
        
        const storeToken = useAuthStore.getState().token;
        const effectiveToken = token || storeToken || (typeof window !== 'undefined' ? localStorage.getItem('token') : null);
        let wsUrl = `${baseUrl}/ws/projects/${projectId}/stream`;
        if (effectiveToken) {
            wsUrl += `?token=${encodeURIComponent(effectiveToken)}`;
        }

        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
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

                // Skip pong messages
                if (update.type === 'pong') return;

                setUpdates((prev) => [...prev.slice(-99), update]); // Keep last 100 updates

                // Helper to update logs
                if (update.message) {
                    setLogs((prev) => [...prev.slice(-49), update.message!]);
                }

                // Helper to map agent names to project store status
                const mapAgentToStatus = (agentName?: string) => {
                    if (!agentName) return;
                    const statusMap: Record<string, ResearchProject['status']> = {
                        supervisor: 'planning',
                        planner: 'planning',
                        discovery: 'searching',
                        retriever: 'searching',
                        ingestion: 'analyzing',
                        matrix_builder: 'analyzing',
                        analyzer: 'analyzing',
                        synthesizer: 'synthesizing',
                        critic: 'synthesizing',
                        auditor: 'synthesizing',
                    };
                    const newStatus = statusMap[agentName.toLowerCase()];
                    if (newStatus && projectId) {
                        updateProjectStatus(projectId, newStatus);
                    }
                };

                switch (update.type) {
                    // --- v3.2 Granular Multi-Agent Event Handlers ---
                    case 'discovery_started':
                        setCurrentAgent(update.agent || 'discovery');
                        mapAgentToStatus('discovery');
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'paper_discovered':
                        setCurrentAgent('discovery');
                        mapAgentToStatus('discovery');
                        setTotalPapers((prev) => prev + 1);
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'pdf_parsed':
                        setCurrentAgent('ingestion');
                        mapAgentToStatus('ingestion');
                        setPapersAnalyzed((prev) => prev + 1);
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'matrix_row_added':
                        setCurrentAgent('matrix_builder');
                        mapAgentToStatus('matrix_builder');
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'thematic_draft_ready':
                        setCurrentAgent('synthesizer');
                        mapAgentToStatus('synthesizer');
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'critic_verdict':
                        setCurrentAgent('critic');
                        mapAgentToStatus('critic');
                        if (update.data) {
                            setLatestCriticVerdict({
                                score: (update.data.score as number) ?? 0,
                                should_refine: Boolean(update.data.should_refine),
                                iteration: (update.data.iteration as number) ?? 0,
                                dimension_scores: update.data.dimension_scores as Record<string, number>,
                                weaknesses: update.data.weaknesses as string[],
                                guidance: update.data.guidance as string,
                            });
                        }
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'fact_checked':
                        setCurrentAgent('auditor');
                        mapAgentToStatus('auditor');
                        if (update.data) {
                            setLatestFactCheck({
                                precision_score: (update.data.precision_score as number) ?? 100,
                                passed: Boolean(update.data.passed),
                                entailed_count: update.data.entailed_count as number,
                                neutral_count: update.data.neutral_count as number,
                                contradiction_count: update.data.contradiction_count as number,
                                total_propositions: update.data.total_propositions as number,
                            });
                        }
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'pipeline_completed':
                    case 'complete':
                        setProgress(100);
                        setCurrentAgent(null);
                        if (projectId) {
                            updateProjectStatus(projectId, 'completed');
                        }
                        queryClient.invalidateQueries({ queryKey: ['project', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['project-report', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['project-matrix', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['project-gaps', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['projects'] });
                        break;

                    case 'pipeline_stopped':
                        setCurrentAgent(null);
                        if (projectId) {
                            updateProjectStatus(projectId, 'stopped');
                        }
                        queryClient.invalidateQueries({ queryKey: ['project', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['projects'] });
                        break;

                    case 'pipeline_error':
                    case 'error':
                        setCurrentAgent(null);
                        if (projectId) {
                            updateProjectStatus(projectId, 'error');
                        }
                        queryClient.invalidateQueries({ queryKey: ['project', projectId] });
                        queryClient.invalidateQueries({ queryKey: ['projects'] });
                        break;

                    // --- Legacy Event Handlers ---
                    case 'agent_started':
                    case 'status':
                        if (update.agent) {
                            setCurrentAgent(update.agent);
                            mapAgentToStatus(update.agent);
                        }
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'agent_completed':
                        break;

                    case 'progress':
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'paper_found':
                        if (update.data?.total) {
                            setTotalPapers(update.data.total as number);
                        }
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'paper_analyzed':
                        if (update.data?.current !== undefined) {
                            setPapersAnalyzed(update.data.current as number);
                        } else {
                            setPapersAnalyzed((prev) => prev + 1);
                        }
                        if (update.data?.total !== undefined) {
                            setTotalPapers(update.data.total as number);
                        }
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;

                    case 'log':
                        if (typeof update.progress === 'number') {
                            setProgress(update.progress);
                        }
                        break;
                }
            } catch (e) {
                console.error('[WebSocket] Failed to parse message:', e);
            }
        };

        ws.onclose = () => {
            setIsConnected(false);
            wsRef.current = null;

            if (pingIntervalRef.current) {
                clearInterval(pingIntervalRef.current);
                pingIntervalRef.current = null;
            }

            // Auto-reconnect logic
            if (autoReconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
                reconnectAttemptsRef.current++;
                reconnectTimeoutRef.current = setTimeout(connect, reconnectDelay);
            }
        };

        ws.onerror = (error) => {
            console.error('[WebSocket] Error:', error);
        };
    }, [projectId, token, autoReconnect, reconnectDelay, maxReconnectAttempts, queryClient, updateProjectStatus]);

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
        papersAnalyzed,
        totalPapers,
        latestCriticVerdict,
        latestFactCheck,
        connect,
        disconnect,
        clearUpdates,
    };
}

export default useProjectStream;