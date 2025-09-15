import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { Container, Box, TextField, Button, Typography, Paper } from '@mui/material';

const RegisterPage = () => {
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const navigate = useNavigate();
    const register = useAuthStore((state) => state.register);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const success = await register({ name, email, password });
        if (success) {
            navigate('/');
        }
    };

    return (
        <Container component="main" maxWidth="xs" className='mt-10'>
            <Paper elevation={6} sx={{ marginTop: 18, padding: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
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
                    <Button
                        type="submit"
                        fullWidth
                        variant="contained"
                        sx={{ mt: 3, mb: 2 }}
                    >
                        Sign Up
                    </Button>
                    <Box textAlign="center">
                        <Link to="/login" style={{ textDecoration: 'none' }}>
                            <Typography variant="body2">
                                {"Already have an account? Sign In"}
                            </Typography>
                        </Link>
                    </Box>
                </Box>
            </Paper>
        </Container>
    );
};

export default RegisterPage;