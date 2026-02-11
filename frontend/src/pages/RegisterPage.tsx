import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import { useBackendWarmup } from "../hooks/useBackendWarmup";
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  Divider,
  CircularProgress,
  IconButton,
  InputAdornment,
  useMediaQuery,
} from "@mui/material";
import { styled } from "@mui/system";
import EditNoteIcon from "@mui/icons-material/EditNote";
import MapIcon from "@mui/icons-material/Map";
import ScienceIcon from "@mui/icons-material/Science";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import Visibility from "@mui/icons-material/Visibility";
import VisibilityOff from "@mui/icons-material/VisibilityOff";

const PageContainer = styled(Box)(() => ({
  minHeight: "100vh",
  width: "100%",
  display: "flex",
  backgroundColor: "#09090B",
  color: "#F4F4F5",
  fontFamily: "'Inter', sans-serif",
  overflow: "hidden",
}));

const FormSection = styled(Box)(({ theme }) => ({
  flex: 1,
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
  padding: "2rem",
  position: "relative",
  backgroundColor: "#09090B",
  zIndex: 10,
  [theme.breakpoints.up("md")]: {
    paddingLeft: "8%", 
    paddingRight: "4%",
    borderRight: "1px solid #27272F",
  },
  [theme.breakpoints.up("lg")]: {
    paddingLeft: "12%", 
  },
}));

const VisualSection = styled(Box)(({ theme }) => ({
  flex: 1,
  display: "none",
  position: "relative",
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
  padding: "4rem",
  background: `
    radial-gradient(circle at 15% 50%, rgba(255, 185, 0, 0.08) 0%, transparent 25%),
    radial-gradient(circle at 85% 30%, rgba(0, 245, 200, 0.08) 0%, transparent 25%),
    linear-gradient(135deg, #09090B 0%, #18181B 100%)
  `,
  [theme.breakpoints.up("md")]: {
    display: "flex",
  },
}));

const NoirTextField = styled(TextField)({
  "& .MuiOutlinedInput-root": {
    backgroundColor: "#18181B",
    borderRadius: "8px",
    color: "#F4F4F5",
    transition: "all 0.2s ease",
    "& fieldset": { borderColor: "#27272F" },
    "&:hover fieldset": { borderColor: "#52525B" },
    "&.Mui-focused fieldset": {
      borderColor: "#FFB900",
      borderWidth: "1px",
      boxShadow: "0 0 0 1px rgba(255, 185, 0, 0.2)",
    },
    "& input": {
      "&:-webkit-autofill": {
        WebkitBoxShadow: "0 0 0 1000px #18181B inset !important",
        WebkitTextFillColor: "#F4F4F5 !important",
        caretColor: "#F4F4F5",
        borderRadius: "inherit",
        transition: "background-color 5000s ease-in-out 0s",
      },
    },
  },
  "& .MuiInputLabel-root": {
    color: "#71717A",
    "&.Mui-focused": { color: "#FFB900" },
  },
});

const GlassCard = styled(Paper)(() => ({
  backgroundColor: "rgba(9, 9, 11, 0.6)",
  backdropFilter: "blur(20px) saturate(180%)",
  WebkitBackdropFilter: "blur(20px) saturate(180%)",
  border: "1px solid rgba(255, 255, 255, 0.08)",
  borderRadius: "12px",
  padding: "1.5rem",
  color: "#F4F4F5",
  transition: "transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.2s ease",
  "&:hover": {
    transform: "translateY(-4px)",
    boxShadow: "0 10px 20px rgba(0,0,0,0.2)",
    border: "1px solid rgba(255, 185, 0, 0.3)",
  },
}));

const RegisterPage: React.FC = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  
  const navigate = useNavigate();
  const { register, loginWithOAuth, isAuthenticated, isInitialized } = useAuthStore();
  const isMdUp = useMediaQuery("(min-width:900px)");
  
  useBackendWarmup();

  useEffect(() => {
    if (isInitialized && isAuthenticated) {
      navigate("/dashboard", { replace: true });
    }
  }, [isInitialized, isAuthenticated, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setLoading(true);
      const success = await register({ name, email, password });
      if (success) navigate("/dashboard");
    } finally {
      setLoading(false);
    }
  };

  const steps = [
    { icon: <EditNoteIcon sx={{ color: "#FFB900" }} />, title: "1. You Ask", desc: "Provide your research question." },
    { icon: <MapIcon sx={{ color: "#00F5C8" }} />, title: "2. Plan & Search", desc: "Agents scour academic sources." },
    { icon: <ScienceIcon sx={{ color: "#00B894" }} />, title: "3. Analyze", desc: "Extraction of key findings." },
    { icon: <AutoAwesomeIcon sx={{ color: "#FFB900" }} />, title: "4. Insight", desc: "Synthesized review delivered." },
  ];

  return (
    <PageContainer>
      <FormSection>
        <Box sx={{ width: "100%", maxWidth: "400px" }}>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              mb: 1,
              background: "linear-gradient(135deg, #FFB900 0%, #00F5C8 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              letterSpacing: "-0.02em",
            }}
          >
            Join Scholar Agent
          </Typography>
          
          <Typography variant="body1" sx={{ color: "#A1A1AA", mb: 4, fontFamily: "'Crimson Pro', serif", fontSize: "1.1rem" }}>
            Begin your research journey with AI-driven depth.
          </Typography>

          <Box component="form" onSubmit={handleSubmit} noValidate>
            <NoirTextField
              margin="normal"
              required
              fullWidth
              label="Full Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={loading}
            />
            <NoirTextField
              margin="normal"
              required
              fullWidth
              label="Email Address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={loading}
            />
            <NoirTextField
              margin="normal"
              required
              fullWidth
              label="Password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton onClick={() => setShowPassword(!showPassword)} edge="end" sx={{ color: "#71717A" }}>
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{
                mt: 4, mb: 2, py: 1.8,
                bgcolor: "#FFB900", color: "#09090B",
                fontWeight: 700, textTransform: "none", fontSize: "1rem", borderRadius: "8px",
                "&:hover": { bgcolor: "#E6A600" },
              }}
            >
              {loading ? <CircularProgress size={24} sx={{ color: "#09090B" }} /> : "Create Account"}
            </Button>

            <Box sx={{ textAlign: "center", mb: 3 }}>
              <Typography variant="body2" sx={{ color: "#71717A" }}>
                Already a researcher?{" "}
                <Link to="/login" style={{ textDecoration: "none" }}>
                  <span style={{ color: "#00F5C8", fontWeight: 600 }}>Sign In</span>
                </Link>
              </Typography>
            </Box>

            <Divider sx={{ my: 3, borderColor: "#27272F" }}>
              <Typography variant="caption" sx={{ color: "#52525B", px: 1 }}>OR</Typography>
            </Divider>

            <Button
              fullWidth
              onClick={() => loginWithOAuth("google")}
              disabled={loading}
              sx={{
                bgcolor: "#18181B", border: "1px solid #3F3F46", color: "#E4E4E7",
                py: 1.2, textTransform: "none", borderRadius: "8px",
                "&:hover": { bgcolor: "#27272F", borderColor: "#52525B" },
              }}
              startIcon={
                <Box component="svg" viewBox="0 0 48 48" width="20px" height="20px">
                  <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z" />
                  <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z" />
                  <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z" />
                  <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z" />
                </Box>
              }
            >
              Continue with Google
            </Button>
          </Box>
        </Box>
      </FormSection>

      {isMdUp && (
        <VisualSection>
          <Box
            sx={{
              position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
              backgroundImage: "linear-gradient(#27272F 1px, transparent 1px), linear-gradient(90deg, #27272F 1px, transparent 1px)",
              backgroundSize: "40px 40px", opacity: 0.1, zIndex: 0, pointerEvents: "none",
            }}
          />

          <Box sx={{ position: "relative", zIndex: 1, maxWidth: "600px", width: "100%" }}>
            <Box sx={{ mb: 6 }}>
              <Typography variant="h2" sx={{ fontWeight: 800, fontSize: "3rem", lineHeight: 1.1, mb: 2, color: "#F4F4F5" }}>
                Research at the <br />
                <Box component="span" sx={{ color: "#FFB900" }}>Speed of Thought.</Box>
              </Typography>
              <Typography variant="h6" sx={{ fontFamily: "'Crimson Pro', serif", color: "#A1A1AA", fontWeight: 400, fontSize: "1.35rem", lineHeight: 1.6 }}>
                Our multi-agent system transforms complex questions into synthesized, actionable insights—fast.
              </Typography>
            </Box>

            <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 3 }}>
              {steps.map((step, index) => (
                <GlassCard key={index} elevation={0}>
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", width: 48, height: 48, borderRadius: "50%", bgcolor: "rgba(255,255,255,0.05)", mb: 2 }}>
                    {step.icon}
                  </Box>
                  <Typography variant="h6" sx={{ fontWeight: 700, fontSize: "1rem", mb: 0.5 }}>{step.title}</Typography>
                  <Typography variant="body2" sx={{ color: "#A1A1AA", fontSize: "0.875rem" }}>{step.desc}</Typography>
                </GlassCard>
              ))}
            </Box>
          </Box>
        </VisualSection>
      )}
    </PageContainer>
  );
};

export default RegisterPage;