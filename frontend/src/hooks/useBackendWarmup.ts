import { useEffect, useState, useRef, useCallback } from 'react';
import axios from 'axios';
import { RENDER_BACKEND_URL } from '../config';

export const useBackendWarmup = () => {
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [warmupAttempts, setWarmupAttempts] = useState(0);
  const mountedRef = useRef(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const pingBackend = useCallback(async () => {
    try {
      await axios.get(`${RENDER_BACKEND_URL}/api/health`, {
        timeout: 5000,
      });
      if (mountedRef.current) {
        setIsBackendReady(true);
        if (intervalRef.current) clearInterval(intervalRef.current);
        console.log('✅ Render backend is ready');
      }
    } catch {
      if (mountedRef.current) {
        setWarmupAttempts(prev => prev + 1);
        console.log('⏳ Warming up backend...');
      }
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    pingBackend();
    
    intervalRef.current = setInterval(pingBackend, 10000);
    
    timeoutRef.current = setTimeout(() => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      console.log('⚠️ Backend warmup timeout - will retry on demand');
    }, 180000);

    return () => {
      mountedRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [pingBackend]);

  return { isBackendReady, warmupAttempts };
};
