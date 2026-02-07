import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import { useBackendWarmup } from "../hooks/useBackendWarmup";
import {
  Container,
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  Avatar,
  useTheme,
  useMediaQuery,
  CircularProgress,
  Divider,
} from "@mui/material";
import EditNoteIcon from "@mui/icons-material/EditNote";
import MapIcon from "@mui/icons-material/Map";
import ScienceIcon from "@mui/icons-material/Science";
import ArticleIcon from "@mui/icons-material/Article";

const StepCard: React.FC<{
  icon: React.ReactNode;
  title: string;
  desc: string;
}> = ({ icon, title, desc }) => {
  return (
    <Paper
      elevation={0}
      sx={{
        textAlign: "center",
        px: 2,
        py: 3,
        height: "100%", // Ensure cards in the same row have the same height
        minHeight: 160,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start",
        position: "relative",
        overflow: "visible",
        filter: "drop-shadow(0px 2px 8px rgba(0,0,0,0.12))",
        borderRadius: 2,
        backgroundColor: "rgba(255,255,255,0.6)",
        backdropFilter: "blur(10px)",
        WebkitBackdropFilter: "blur(10px)",
        transition: "transform 200ms ease, box-shadow 200ms ease",
        "&:hover": {
          transform: "translateY(-6px)",
          boxShadow: "0 10px 30px rgba(0,0,0,0.08)",
        },
      }}
    >
      <Avatar sx={{ bgcolor: "primary.main", width: 56, height: 56, mb: 2 }}>
        {icon}
      </Avatar>
      <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>
        {title}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 320 }}>
        {desc}
      </Typography>
    </Paper>
  );
};

const steps = [
  {
    icon: <EditNoteIcon fontSize="large" />,
    title: "1. You Ask",
    desc: "Provide your research question and title — our system begins immediately.",
  },
  {
    icon: <MapIcon fontSize="large" />,
    title: "2. We Plan & Search",
    desc: "Planner creates a strategy and scours top academic sources for relevant papers.",
  },
  {
    icon: <ScienceIcon fontSize="large" />,
    title: "3. Agents Analyze",
    desc: "Specialized agents read each paper extracting findings, methods, and limitations.",
  },
  {
    icon: <ArticleIcon fontSize="large" />,
    title: "4. You Get Insights",
    desc: "A synthesized literature review is produced and delivered to your email.",
  },
];

const RegisterPage: React.FC = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [oauthLoading, setOauthLoading] = useState(false);
  const navigate = useNavigate();
  const { register, loginWithOAuth, isAuthenticated, isInitialized } = useAuthStore();
  useBackendWarmup();

  useEffect(() => {
    if (isInitialized && isAuthenticated) {
      navigate("/dashboard", { replace: true });
    }
  }, [isInitialized, isAuthenticated, navigate]);

  const theme = useTheme();
  const isMdUp = useMediaQuery(theme.breakpoints.up("md"));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setLoading(true);
      const success = await register({ name, email, password });
      if (success) {
        navigate("/dashboard");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    setOauthLoading(true);
    await loginWithOAuth("google");
  };

  // SMALL SCREENS: render original unchanged UI
  if (!isMdUp) {
    return (
      <Container component="main" maxWidth="xs" className="mt-10">
        <Paper
          elevation={6}
          sx={{
            marginTop: 18,
            padding: 4,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <Typography component="h1" variant="h5">
            Sign up
          </Typography>
          <Box
            component="form"
            onSubmit={handleSubmit}
            noValidate
            sx={{ mt: 1 }}
          >
            <TextField
              margin="normal"
              required
              fullWidth
              id="name"
              label="Full Name"
              name="name"
              autoComplete="name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={loading}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={loading}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2 }}
              disabled={loading}
            >
              {loading ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                "Sign Up"
              )}
            </Button>
            <Box textAlign="center">
              <Link to="/login" style={{ textDecoration: "none" }}>
                <Typography variant="body2">
                  {"Already have an account? "}
                  <span className="text-teal-500 font-bold">Sign In</span>
                </Typography>
              </Link>
            </Box>
            <Divider sx={{ my: 2 }}>or</Divider>

            <button
              className="
                flex items-center justify-center
                bg-white border border-[#747775] 
                rounded-md px-3 py-2 
                hover:bg-[#F8FAFF] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#4285F4]
                transition-colors duration-200
                min-w-[200px] h-10
                mx-auto
              "
              style={{ fontFamily: "'Roboto', sans-serif", fontWeight: 500 }}
              onClick={handleGoogleSignUp}
              disabled={oauthLoading || loading}
            >
              <div className="flex items-center justify-center w-full">
                <div className="mr-3 w-[18px] h-[18px]">
                  <svg
                    version="1.1"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 48 48"
                    className="block w-full h-full"
                  >
                    <path
                      fill="#EA4335"
                      d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
                    ></path>
                    <path
                      fill="#4285F4"
                      d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"
                    ></path>
                    <path
                      fill="#FBBC05"
                      d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"
                    ></path>
                    <path
                      fill="#34A853"
                      d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"
                    ></path>
                    <path fill="none" d="M0 0h48v48H0z"></path>
                  </svg>
                </div>

                <span className="text-[#1F1F1F] text-sm tracking-wide">
                  Continue with Google
                </span>
              </div>
            </button>
          </Box>
        </Paper>
      </Container>
    );
  }

  // MD+ : two-column layout using Box with Flexbox
  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {/* LEFT: form centered vertically */}
        <Box sx={{ width: { xs: "100%", md: "50%", lg: "41.67%" }, p: 3 }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              height: "100%",
              minHeight: 480,
            }}
          >
            <Paper
              elevation={6}
              sx={{
                width: "100%",
                p: 6,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
              }}
            >
              <Typography component="h1" variant="h5">
                Sign up
              </Typography>
              <Box
                component="form"
                onSubmit={handleSubmit}
                noValidate
                sx={{ mt: 1, width: "100%" }}
              >
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="name"
                  label="Full Name"
                  name="name"
                  autoComplete="name"
                  autoFocus
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  id="email"
                  label="Email Address"
                  name="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
                <TextField
                  margin="normal"
                  required
                  fullWidth
                  name="password"
                  label="Password"
                  type="password"
                  id="password"
                  autoComplete="new-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <Button
                  type="submit"
                  fullWidth
                  variant="contained"
                  sx={{ mt: 3, mb: 2 }}
                  disabled={loading}
                >
                  {loading ? (
                    <CircularProgress size={20} color="inherit" />
                  ) : (
                    "Sign Up"
                  )}
                </Button>
                <Box textAlign="center">
                  <Link to="/login" style={{ textDecoration: "none" }}>
                    <Typography variant="body2">
                      {"Already have an account? "}
                      <span className="text-teal-500 font-bold">Sign In</span>
                    </Typography>
                  </Link>
                </Box>
                <Divider sx={{ my: 2 }}>or</Divider>

                <button
                  className="
                    flex items-center justify-center
                    bg-white border border-[#747775] 
                    rounded-md px-3 py-2 
                    hover:bg-[#F8FAFF] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#4285F4]
                    transition-colors duration-200
                    min-w-[200px] h-10
                    mx-auto
                    
                  "
                  onClick={handleGoogleSignUp}
                  disabled={oauthLoading || loading}
                  style={{ fontFamily: "'Roboto', sans-serif", fontWeight: 500 }}
                >
                  <div className="flex items-center justify-center w-full">
                    <div className="mr-3 w-[18px] h-[18px]">
                      <svg
                        version="1.1"
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 48 48"
                        className="block w-full h-full"
                      >
                        <path
                          fill="#EA4335"
                          d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
                        ></path>
                        <path
                          fill="#4285F4"
                          d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"
                        ></path>
                        <path
                          fill="#FBBC05"
                          d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"
                        ></path>
                        <path
                          fill="#34A853"
                          d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"
                        ></path>
                        <path fill="none" d="M0 0h48v48H0z"></path>
                      </svg>
                    </div>

                    <span className="text-[#1F1F1F] text-sm tracking-wide">
                      Continue with Google
                    </span>
                  </div>
                </button>
                <Box textAlign="center" sx={{ mt: 1 }}></Box>
              </Box>
            </Paper>
          </Box>
        </Box>

        {/* RIGHT: KnowPage-inspired panel */}
        <Box sx={{ width: { xs: "100%", md: "50%", lg: "58.33%" }, p: 3 }}>
          <Box
            sx={{
              px: { md: 4 },
              py: { md: 4 },
            }}
          >
            <Box sx={{ textAlign: "center" }}>
              <Typography
                component="h2"
                sx={{
                  fontWeight: 800,
                  fontSize: { md: "1.8rem", lg: "2.2rem" },
                  lineHeight: 1.05,
                  background: "linear-gradient(90deg,#0ea5a4,#2563eb)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  mb: 1,
                }}
              >
                Your Intelligent Research Partner
              </Typography>

              <Typography
                variant="body1"
                color="text.secondary"
                sx={{ mb: 3, maxWidth: 680 }}
              >
                Discover how our multi-agent system transforms complex research
                questions into synthesized, actionable insights — fast.
              </Typography>
            </Box>

            <Box sx={{ display: "flex", flexWrap: "wrap" }}>
              {steps.map((s, idx) => (
                <Box
                  key={idx}
                  sx={{ width: { xs: "100%", sm: "50%" }, p: 1.5 }}
                >
                  <StepCard icon={s.icon} title={s.title} desc={s.desc} />
                </Box>
              ))}
            </Box>
          </Box>
        </Box>
      </Box>
    </Container>
  );
};

export default RegisterPage;
