import React from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';

interface ExampleOutputPreviewProps {
  title: string;
  excerpt: string;
  fullExampleLink?: string;
  remotionVideoSrc?: string; // Optional for dynamic previews
}

const ExampleOutputPreview: React.FC<ExampleOutputPreviewProps> = ({
  title,
  excerpt,
  fullExampleLink,
  remotionVideoSrc,
}) => {

  return (
    <Paper
      sx={{
        p: 4,
        borderRadius: 'var(--radius-lg)',
        // Dark mode glassmorphism effect
        backgroundColor: 'rgba(9, 9, 11, 0.7)', // Obsidian 900 with transparency
        backdropFilter: 'blur(20px) saturate(180%)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        minHeight: 250,
        my: 4,
        transition: 'transform 0.3s ease, box-shadow 0.3s ease',
        '&:hover': {
          transform: 'translateY(-5px)',
          boxShadow: '0 12px 40px 0 rgba(0, 0, 0, 0.5)',
        },
      }}
    >
      <Box>
        <Typography variant="h5" gutterBottom component="h4" sx={{ color: 'var(--text-primary)' }}>
          {title}
        </Typography>
        <Typography variant="body1" sx={{ color: 'var(--text-secondary)' }} paragraph>
          {excerpt}
        </Typography>
        {remotionVideoSrc && (
          <Box
            sx={{
              mt: 2,
              height: 150,
              bgcolor: 'var(--bg-elevated)',
              borderRadius: 'var(--radius-md)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              overflow: 'hidden',
            }}
          >
            <video
              autoPlay
              loop
              muted
              playsInline
              src={remotionVideoSrc}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            >
              Your browser does not support the video tag.
            </video>
          </Box>
        )}
      </Box>
      {fullExampleLink && (
        <Button 
          variant="text" 
          sx={{ 
            mt: 2, 
            alignSelf: 'flex-start', 
            color: 'var(--accent-primary)',
            '&:hover': {
              color: 'var(--accent-secondary)',
              backgroundColor: 'rgba(255, 185, 0, 0.1)',
            }
          }} 
          href={fullExampleLink} 
          target="_blank" 
          rel="noopener noreferrer"
        >
          See Full Example
        </Button>
      )}
    </Paper>
  );
};

export default ExampleOutputPreview;
