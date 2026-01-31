// Tests for useProjectStream hook

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactNode } from "react";
import { useProjectStream, AgentUpdate } from "./useProjectStream";

// Mock WebSocket with static OPEN/CLOSED constants
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static OPEN = 1;
  static CLOSED = 3;
  static CONNECTING = 0;
  static CLOSING = 2;

  url: string;
  readyState: number = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: ((event: { code: number; reason: string }) => void) | null = null;
  onerror: ((error: unknown) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    // Simulate async connection
    Promise.resolve().then(() => {
      if (this.readyState === MockWebSocket.CONNECTING) {
        this.readyState = MockWebSocket.OPEN;
        this.onopen?.();
      }
    });
  }

  send(_data: string) {
    // Mock send
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    // Pass a proper CloseEvent-like object
    this.onclose?.({ code: 1000, reason: "Normal closure" });
  }

  // Helper to simulate receiving a message
  simulateMessage(data: AgentUpdate) {
    this.onmessage?.({ data: JSON.stringify(data) });
  }

  // Helper to simulate error
  simulateError(error: unknown) {
    this.onerror?.(error);
  }
}

// Store original WebSocket
const OriginalWebSocket = globalThis.WebSocket;

// Install mock WebSocket globally BEFORE tests run
beforeEach(() => {
  MockWebSocket.instances = [];
  // @ts-expect-error - Mocking WebSocket with compatible interface
  globalThis.WebSocket = MockWebSocket;
});

afterEach(() => {
  globalThis.WebSocket = OriginalWebSocket;
});

describe("useProjectStream", () => {
  let queryClient: QueryClient;

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("connection management", () => {
    it("should not connect when projectId is undefined", () => {
      const { result } = renderHook(() => useProjectStream(undefined), {
        wrapper,
      });

      expect(result.current.isConnected).toBe(false);
      expect(MockWebSocket.instances).toHaveLength(0);
    });

    it("should connect when projectId is provided", async () => {
      const { result } = renderHook(() => useProjectStream("project-123"), {
        wrapper,
      });

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1);
      });

      expect(MockWebSocket.instances[0].url).toContain("project-123");
    });

    it("should set isConnected to true after connection", async () => {
      const { result } = renderHook(() => useProjectStream("project-123"), {
        wrapper,
      });

      await waitFor(() => {
        expect(result.current.isConnected).toBe(true);
      });
    });

    it("should disconnect when unmounted", async () => {
      const { result, unmount } = renderHook(
        () => useProjectStream("project-123"),
        { wrapper },
      );

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1);
      });

      unmount();

      expect(MockWebSocket.instances[0].readyState).toBe(3); // CLOSED
    });
  });

  describe("message handling", () => {
    it("should track current agent from status updates", async () => {
      const { result } = renderHook(() => useProjectStream("project-123"), {
        wrapper,
      });

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1);
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({
          type: "agent_started",
          agent: "planner",
          message: "Starting planning phase",
        });
      });

      expect(result.current.currentAgent).toBe("planner");
    });

    it("should update progress from progress events", async () => {
      const { result } = renderHook(() => useProjectStream("project-123"), {
        wrapper,
      });

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1);
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({
          type: "progress",
          progress: 50,
          message: "Halfway done",
        });
      });

      expect(result.current.progress).toBe(50);
    });

    it("should accumulate updates", async () => {
      const { result } = renderHook(() => useProjectStream("project-123"), {
        wrapper,
      });

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1);
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({
          type: "agent_started",
          agent: "planner",
        });
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({
          type: "progress",
          progress: 25,
        });
      });

      expect(result.current.updates).toHaveLength(2);
    });

    it("should collect log messages", async () => {
      const { result } = renderHook(() => useProjectStream("project-123"), {
        wrapper,
      });

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1);
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({
          type: "log",
          message: "Processing paper 1 of 10",
        });
      });

      expect(result.current.logs).toContain("Processing paper 1 of 10");
    });
  });

  describe("clearUpdates", () => {
    it("should clear all updates when called", async () => {
      const { result } = renderHook(() => useProjectStream("project-123"), {
        wrapper,
      });

      await waitFor(() => {
        expect(MockWebSocket.instances).toHaveLength(1);
      });

      act(() => {
        MockWebSocket.instances[0].simulateMessage({
          type: "progress",
          progress: 25,
        });
      });

      expect(result.current.updates).toHaveLength(1);

      act(() => {
        result.current.clearUpdates();
      });

      expect(result.current.updates).toHaveLength(0);
    });
  });
});
