import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import {
  Container,
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  // Grid has been removed from imports
  Avatar,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import EditNoteIcon from '@mui/icons-material/EditNote';
import MapIcon from '@mui/icons-material/Map';
import ScienceIcon from '@mui/icons-material/Science';
import ArticleIcon from '@mui/icons-material/Article';

const StepCard: React.FC<{ icon: React.ReactNode; title: string; desc: string }> = ({ icon, title, desc }) => {
  return (
    <Paper
      elevation={0}
      sx={{
        textAlign: 'center',
        px: 2,
        py: 3,
        height: '100%', // Ensure cards in the same row have the same height
        minHeight: 160,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'flex-start',
        position: 'relative',
        overflow: 'visible',
        filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.12))',
        borderRadius: 2,
        backgroundColor: 'rgba(255,255,255,0.6)',
        backdropFilter: 'blur(10px)',
        WebkitBackdropFilter: 'blur(10px)',
        transition: 'transform 200ms ease, box-shadow 200ms ease',
        '&:hover': {
          transform: 'translateY(-6px)',
          boxShadow: '0 10px 30px rgba(0,0,0,0.08)',
        },
      }}
    >
      <Avatar sx={{ bgcolor: 'primary.main', width: 56, height: 56, mb: 2 }}>{icon}</Avatar>
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
    title: '1. You Ask',
    desc: 'Provide your research question and title — our system begins immediately.',
  },
  {
    icon: <MapIcon fontSize="large" />,
    title: '2. We Plan & Search',
    desc: 'Planner creates a strategy and scours top academic sources for relevant papers.',
  },
  {
    icon: <ScienceIcon fontSize="large" />,
    title: '3. Agents Analyze',
    desc: 'Specialized agents read each paper extracting findings, methods, and limitations.',
  },
  {
    icon: <ArticleIcon fontSize="large" />,
    title: '4. You Get Insights',
    desc: 'A synthesized literature review is produced and delivered to your email.',
  },
];

const RegisterPage: React.FC = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  const register = useAuthStore((state) => state.register);

  const theme = useTheme();
  const isMdUp = useMediaQuery(theme.breakpoints.up('md'));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const success = await register({ name, email, password });
    if (success) {
      navigate('/');
    }
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
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
          }}
        >
          <Typography component="h1" variant="h5">
            Sign up
          </Typography>
          <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
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
            <Button type="submit" fullWidth variant="contained" sx={{ mt: 3, mb: 2 }}>
              Sign Up
            </Button>
            <Box textAlign="center">
              <Link to="/login" style={{ textDecoration: 'none' }}>
                <Typography variant="body2">{"Already have an account? Sign In"}</Typography>
              </Link>
            </Box>
          </Box>
        </Paper>
      </Container>
    );
  }

  // MD+ : two-column layout using Box with Flexbox
  return (
    <Container maxWidth="lg" sx={{ py: 8 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center' }}>
        {/* LEFT: form centered vertically */}
        <Box sx={{ width: { xs: '100%', md: '50%', lg: '41.67%' }, p: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', height: '100%', minHeight: 480 }}>
            <Paper
              elevation={6}
              sx={{
                width: '100%',
                p: 6,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
              }}
            >
              <Typography component="h1" variant="h5">
                Sign up
              </Typography>
              <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1, width: '100%' }}>
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
                <Button type="submit" fullWidth variant="contained" sx={{ mt: 3, mb: 2 }}>
                  Sign Up
                </Button>
                <Box textAlign="center">
                  <Link to="/login" style={{ textDecoration: 'none' }}>
                    <Typography variant="body2">{"Already have an account? Sign In"}</Typography>
                  </Link>
                </Box>
              </Box>
            </Paper>
          </Box>
        </Box>

        {/* RIGHT: KnowPage-inspired panel */}
        <Box sx={{ width: { xs: '100%', md: '50%', lg: '58.33%' }, p: 3 }}>
          <Box
            sx={{
              px: { md: 4 },
              py: { md: 4 },
            }}
          >
            <Box sx={{textAlign:'center'}}>
            <Typography
              component="h2"
              sx={{
                fontWeight: 800,
                fontSize: { md: '1.8rem', lg: '2.2rem' },
                lineHeight: 1.05,
                background: 'linear-gradient(90deg,#0ea5a4,#2563eb)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 1,
              }}
            >
              Your Intelligent Research Partner
            </Typography>
            
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3, maxWidth: 680 }}>
              Discover how our multi-agent system transforms complex research questions into synthesized,
              actionable insights — fast.
            </Typography>
            </Box>

            <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
              {steps.map((s, idx) => (
                <Box key={idx} sx={{ width: { xs: '100%', sm: '50%' }, p: 1.5 }}>
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