import React, { useState } from 'react';
import { Modal, Box, TextField, Button, Typography, CircularProgress } from '@mui/material';
import { useProjectStore } from '../../store/projectStore';
import { useBackendWarmup } from '../../hooks/useBackendWarmup';

interface CreateProjectModalProps {
    open: boolean;
    onClose: () => void;
}

const style = {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: '90%',
    maxWidth: 600,
    bgcolor: 'rgba(255, 255, 255, 0.7)',
    backdropFilter: 'blur(20px)',
    border: '1px solid rgba(0, 0, 0, 0.1)',
    boxShadow: 24,
    p: 4,
    borderRadius: 4,
};


const CreateProjectModal: React.FC<CreateProjectModalProps> = ({ open, onClose }) => {
    const [title, setTitle] = useState('');
    const [researchQuestion, setResearchQuestion] = useState('');
    const { addProject, isLoading: isCreating } = useProjectStore();
    const { isBackendReady } = useBackendWarmup();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!isBackendReady) return;
        
        addProject({ title, research_question: researchQuestion });
        onClose();
        setTitle('');
        setResearchQuestion('');
    };

    return (
        <Modal open={open} onClose={onClose}>
            <Box sx={style} component="form" onSubmit={handleSubmit} className="space-y-6">
                <Typography variant="h5" component="h2" className="font-bold text-slate-800">
                    Create New Research Project
                </Typography>
                
                {!isBackendReady && (
                   <Box sx={{ bgcolor: 'rgba(255, 165, 0, 0.1)', p: 2, borderRadius: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
                       <CircularProgress size={20} color="warning" />
                       <Typography variant="body2" color="warning.main">
                           Agent backend is waking up (Render free tier). Please wait a moment to create your project.
                       </Typography>
                   </Box>
                )}

                <TextField
                    fullWidth
                    label="Project Title"
                    variant="outlined"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    required
                />
                <TextField
                    fullWidth
                    label="Research Question"
                    variant="outlined"
                    multiline
                    rows={3}
                    value={researchQuestion}
                    onChange={(e) => setResearchQuestion(e.target.value)}
                    required
                />
                <Box className="flex justify-end pt-4 gap-2">
                    <Button onClick={onClose} color="inherit" className='hover:bg-grey-800'>
                        Cancel
                    </Button>
                    <Button 
                        type="submit" 
                        variant="contained" 
                        color="primary" 
                        disabled={isCreating || !isBackendReady} 
                        className='bg-gradient-to-r from-blue-600 to-teal-500 hover:bg-blue-700 text-white'
                    >
                        {isCreating ? 'Creating...' : !isBackendReady ? 'Waking up...' : 'Create Project'}
                    </Button>
                </Box>
            </Box>
        </Modal>
    );
};

export default CreateProjectModal;
