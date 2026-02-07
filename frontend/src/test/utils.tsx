// Test utilities and mock providers for React testing

import { ReactElement, ReactNode } from "react";
import { render, RenderOptions } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

// Create a new query client for each test
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

interface WrapperProps {
  children: ReactNode;
}

// Custom render that includes all providers
function customRender(
  ui: ReactElement,
  options?: Omit<RenderOptions, "wrapper">,
) {
  const queryClient = createTestQueryClient();

  const AllTheProviders = ({ children }: WrapperProps) => {
    return (
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>{children}</BrowserRouter>
      </QueryClientProvider>
    );
  };

  return {
    ...render(ui, { wrapper: AllTheProviders, ...options }),
    queryClient,
  };
}

// Re-export everything
// eslint-disable-next-line react-refresh/only-export-components
export * from "@testing-library/react";
export { customRender as render };
