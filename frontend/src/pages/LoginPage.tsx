import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { Container, Box, TextField, Button, Typography, Paper } from '@mui/material';

const LoginPage = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();
    const login = useAuthStore((state) => state.login);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const success = await login({ username: email, password });
        if (success) {
            navigate('/');
        }
    };

    return (
        <Container component="main" maxWidth="xs" >
            <Paper elevation={6} sx={{ marginTop: 18, padding: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <Typography component="h1" variant="h5">
                    Sign in
                </Typography>
                <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
                    <TextField
                        margin="normal"
                        required
                        fullWidth
                        id="email"
                        label="Email Address"
                        name="email"
                        autoComplete="email"
                        autoFocus
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
                        autoComplete="current-password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />
                    <Button
                        type="submit"
                        fullWidth
                        variant="contained"
                        sx={{ mt: 3, mb: 2 }}
                    >
                        Sign In
                    </Button>
                    <Box textAlign="center">
                        <Link to="/register" style={{ textDecoration: 'none' }}>
                            <Typography variant="body2">
                                {"Don't have an account? Sign Up"}
                            </Typography>
                        </Link>
                    </Box>
                </Box>
            </Paper>
        </Container>
    );
};

export default LoginPage;