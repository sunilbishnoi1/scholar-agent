import { Navigate, Outlet } from "react-router-dom";
import { useAuthStore } from "../../store/authStore";
import { CircularProgress, Box } from "@mui/material";

const ProtectedRoute = () => {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const user = useAuthStore((state) => state.user);

  // If we have a token (isAuthenticated) but the user profile hasn't loaded yet,
  // show a small loading indicator instead of immediately rendering protected routes.
  if (isAuthenticated && !user) {
    return (
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "60vh",
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  return isAuthenticated && user ? (
    <Outlet />
  ) : (
    <Navigate to="/register" replace />
  );
};

export default ProtectedRoute;
