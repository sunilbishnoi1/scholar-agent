import { useState } from 'react';
import { Container, Typography, Box, Paper, Switch, FormControlLabel, Divider } from '@mui/material';
import MarkEmailReadIcon from '@mui/icons-material/MarkEmailRead';

const ToolsPage = () => {
    const [emailNotifications, setEmailNotifications] = useState(true);

    const handleEmailNotificationChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setEmailNotifications(event.target.checked);
        // Here you would typically call an API to save this preference
        console.log('Email notification preference saved:', event.target.checked);
    };

    return (
        <Container maxWidth="md" className="mt-10 pt-10 sm:pt-12">
            <Box className="mb-8">
                <Typography 
                    variant="h4" 
                    component="h1" 
                    className="bg-gradient-to-r from-blue-600 to-teal-500 bg-clip-text text-transparent font-bold mb-2"
                >
                    Integrations & Tools
                </Typography>
                <Typography variant="subtitle1" className="text-slate-500">
                    Connect Scholar Agent with your favorite tools to streamline your research workflow.
                </Typography>
            </Box>

            <Paper 
                elevation={0} 
                sx={{ 
                    p: 4, 
                    borderRadius: '12px', 
                    border: '1px solid',
                    borderColor: 'divider',
                    background: 'rgba(255, 255, 255, 0.5)',
                    backdropFilter: 'blur(10px)',
                    WebkitBackdropFilter: 'blur(10px)',
                }}
            >
                <Box className="flex items-start mb-4">
                    <MarkEmailReadIcon color="primary" sx={{ fontSize: 40, mr: 2, color: '#2563eb' }} />
                    <Box>
                        <Typography variant="h6" component="h2" fontWeight="bold">
                            Email Notifications
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Manage email alerts for your research projects.
                        </Typography>
                    </Box>
                </Box>
                
                <Divider sx={{ my: 2 }} />

                <Box className="flex items-center justify-between">
                    <Typography variant="body1">
                        Receive an email when your report is ready
                    </Typography>
                    <FormControlLabel
                        control={
                            <Switch
                                checked={emailNotifications}
                                onChange={handleEmailNotificationChange}
                                name="emailNotifications"
                                color="primary"
                            />
                        }
                        label={emailNotifications ? 'Enabled' : 'Disabled'}
                        labelPlacement="start"
                        sx={{ ml: 0 }}
                    />
                </Box>
            </Paper>

            {/* You can add more tool integration cards here in the future */}
            {/* 
            <Paper elevation={0} sx={{ mt: 4, ... }}>
                ... another tool ...
            </Paper>
            */}

        </Container>
    );
};

export default ToolsPage;