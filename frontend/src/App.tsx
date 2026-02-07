import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import { CssBaseline } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import Header from "./components/common/Header";
import { LandingNavigation } from "./components/landing/LandingNavigation";
import LandingPage from "./pages/LandingPage";
import DashboardPage from "./pages/DashboardPage";
import ProjectDetailsPage from "./pages/ProjectDetailsPage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import AuthCallbackPage from "./pages/AuthCallbackPage";
import ProtectedRoute from "./components/common/ProtectedRoute";
import { useAuthStore } from "./store/authStore";
import ToolsPage from "./pages/ToolsPage";
import HowItWorksPage from "./pages/HowItWorksPage";
import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { RENDER_BACKEND_URL } from "./config";

const warmupBackend = () => {
  console.log("Warming up backend...");
  fetch(`${RENDER_BACKEND_URL}/api/health`).catch(() => {
    // Ignore errors, we just want to trigger the wake up
  });
};

const queryClient = new QueryClient();

function AppContent() {
  const location = useLocation();
  const isLandingPage = location.pathname === "/";

  return (
    <>
      {isLandingPage ? <LandingNavigation /> : <Header />}
      <main>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/auth/callback" element={<AuthCallbackPage />} />
          <Route path="/how-it-works" element={<HowItWorksPage />} />
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/tools" element={<ToolsPage />} />
            <Route
              path="/project/:projectId"
              element={<ProjectDetailsPage />}
            />
          </Route>
        </Routes>
      </main>
    </>
  );
}

function App() {
  const { token, fetchUser } = useAuthStore();

  useEffect(() => {
    warmupBackend();
    if (token) {
      fetchUser();
    }
  }, [token, fetchUser]);

  return (
    <QueryClientProvider client={queryClient}>
      <CssBaseline />
      <Router>
        <AppContent />
        <ToastContainer
          position="bottom-right"
          autoClose={3000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="dark"
        />
      </Router>
    </QueryClientProvider>
  );
}

export default App;
