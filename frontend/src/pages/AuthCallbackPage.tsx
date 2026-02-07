import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Container, Box, CircularProgress, Typography } from "@mui/material";
import { useAuthStore } from "../store/authStore";
import { neonAuth } from "../api/neonClient";

const AuthCallbackPage: React.FC = () => {
  const navigate = useNavigate();
  const { fetchUser } = useAuthStore();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handleCallback = async () => {
      try {
        const { data, error: sessionError } = await neonAuth.getSession();

        if (sessionError) {
          throw sessionError;
        }

        if (data?.session) {
          useAuthStore.setState({
            token: data.session.access_token,
            isAuthenticated: true,
          });
          await fetchUser();
          navigate("/dashboard", { replace: true });
        } else {
          setError("No session found. Please try logging in again.");
          setTimeout(() => navigate("/login", { replace: true }), 3000);
        }
      } catch (err) {
        console.error("OAuth callback error:", err);
        setError("Authentication failed. Please try again.");
        setTimeout(() => navigate("/login", { replace: true }), 3000);
      }
    };

    handleCallback();
  }, [fetchUser, navigate]);

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "80vh",
          gap: 2,
        }}
      >
        {error ? (
          <>
            <Typography color="error" variant="h6">
              {error}
            </Typography>
            <Typography color="text.secondary">
              Redirecting to login...
            </Typography>
          </>
        ) : (
          <>
            <CircularProgress size={48} />
            <Typography variant="h6">Completing sign in...</Typography>
            <Typography color="text.secondary">
              Please wait while we verify your credentials.
            </Typography>
          </>
        )}
      </Box>
    </Container>
  );
};

export default AuthCallbackPage;
