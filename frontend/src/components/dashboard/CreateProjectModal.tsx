import React, { useState } from 'react';
import { 
    Modal, 
    Box, 
    Typography, 
    CircularProgress, 
    Backdrop, 
    Fade,
    IconButton,
    TextField
} from '@mui/material';
import { styled } from '@mui/system';
import CloseIcon from '@mui/icons-material/Close';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useProjectStore } from '../../store/projectStore';
import { useBackendWarmup } from '../../hooks/useBackendWarmup';

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
    "& input, & textarea": {
      "&:-webkit-autofill": {
        WebkitBoxShadow: "0 0 0 1000px #18181B inset !important",
        WebkitTextFillColor: "#F4F4F5 !important",
        caretColor: "#F4F4F5",
        borderRadius: "inherit", 
      },
    },
  },
  "& .MuiInputLabel-root": {
    color: "#71717A",
    "&.Mui-focused": { color: "#FFB900" },
  },
});

const StyledModal = styled(Modal)({
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
});

const ModalBox = styled(Box)(({ theme }) => ({
    position: 'relative',
    width: '95%',
    maxWidth: '550px',
    backgroundColor: '#18181B', 
    backgroundImage: 'radial-gradient(circle at top right, rgba(255, 185, 0, 0.05), transparent 250px)',
    border: '1px solid #27272F',
    borderRadius: '16px',
    boxShadow: '0 24px 48px rgba(0, 0, 0, 0.4)',
    padding: '2.5rem',
    outline: 'none',
    [theme.breakpoints.down('sm')]: {
        padding: '1.5rem',
    },
}));

const NoirInput = styled(NoirTextField)({
    marginBottom: '1.5rem',
    '& .MuiOutlinedInput-root': {
        '& textarea': {
            fontFamily: "'Crimson Pro', serif",
            fontSize: '1.1rem',
            lineHeight: 1.5,
        }
    }
});

const ActionButton = styled('button')<{ disabled?: boolean; secondary?: boolean }>(({ disabled, secondary }) => ({
    padding: '12px 24px',
    borderRadius: '8px',
    fontSize: '0.95rem',
    fontWeight: 700,
    cursor: disabled ? 'not-allowed' : 'pointer',
    transition: 'all 0.2s ease',
    border: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '8px',
    backgroundColor: secondary ? 'transparent' : '#FFB900',
    color: secondary ? '#A1A1AA' : '#09090B',
    opacity: disabled ? 0.5 : 1,
    '&:hover': {
        backgroundColor: secondary ? 'rgba(255,255,255,0.05)' : '#E6A600',
        color: secondary ? '#F4F4F5' : '#09090B',
        transform: disabled ? 'none' : 'translateY(-1px)',
    },
}));

interface CreateProjectModalProps {
    open: boolean;
    onClose: () => void;
}

const CreateProjectModal: React.FC<CreateProjectModalProps> = ({ open, onClose }) => {
    const [title, setTitle] = useState('');
    const [researchQuestion, setResearchQuestion] = useState('');
    const { addProject, isLoading: isCreating } = useProjectStore();
    const { isBackendReady } = useBackendWarmup();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!isBackendReady || isCreating) return;

        try {
            await addProject({
                title,
                research_question: researchQuestion, 
            });
            
            setTitle('');
            setResearchQuestion('');
            onClose();
        } catch (error) {
            console.error("Failed to create project", error);
        }
    };

    return (
        <StyledModal
            open={open}
            onClose={onClose}
            closeAfterTransition
            slots={{ backdrop: Backdrop }}
            slotProps={{
                backdrop: {
                    timeout: 500,
                    style: { backgroundColor: 'rgba(9, 9, 11, 0.8)', backdropFilter: 'blur(8px)' }
                },
            }}
        >
            <Fade in={open}>
                <ModalBox>
                    <IconButton 
                        onClick={onClose}
                        sx={{ position: 'absolute', top: 16, right: 16, color: '#52525B' }}
                    >
                        <CloseIcon fontSize="small" />
                    </IconButton>

                    <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <AutoAwesomeIcon sx={{ color: '#FFB900', fontSize: '1.8rem' }} />
                        <Box>
                            <Typography variant="h5" sx={{ 
                                fontWeight: 800, 
                                color: '#F4F4F5',
                                letterSpacing: '-0.02em'
                            }}>
                                New Research Project
                            </Typography>
                            <Typography variant="body2" sx={{ color: '#71717A', fontFamily: "'Crimson Pro', serif" }}>
                                Define your hypothesis to deploy the agent swarm.
                            </Typography>
                        </Box>
                    </Box>

                    <form onSubmit={handleSubmit}>
                        <NoirInput
                            fullWidth
                            label="Project Title"
                            placeholder="e.g., The Impact of CRISPR on Crop Resilience"
                            value={title}
                            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTitle(e.target.value)}
                            required
                            disabled={isCreating}
                        />

                        <NoirInput
                            fullWidth
                            label="Research Question"
                            placeholder="State your specific question or thesis..."
                            multiline
                            rows={4}
                            value={researchQuestion}
                            onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setResearchQuestion(e.target.value)}
                            required
                            disabled={isCreating}
                        />

                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 2 }}>
                            <ActionButton type="button" secondary onClick={onClose} disabled={isCreating}>
                                Cancel
                            </ActionButton>
                            
                            <ActionButton 
                                type="submit" 
                                disabled={isCreating || !isBackendReady || !title || !researchQuestion}
                            >
                                {isCreating ? (
                                    <CircularProgress size={20} sx={{ color: '#09090B' }} />
                                ) : !isBackendReady ? (
                                    'Waking up backend...'
                                ) : (
                                    'Initiate Research'
                                )}
                            </ActionButton>
                        </Box>
                    </form>
                </ModalBox>
            </Fade>
        </StyledModal>
    );
};

export default CreateProjectModal;