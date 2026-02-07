import React from 'react';
import { Box, Typography } from '@mui/material';
import { Player } from '@remotion/player';
import { ResearchGapComposition } from './remotion/ResearchGapComposition';

interface ResearchGapDiagramProps {
  remotionVideoSrc?: string; // Optional: path to pre-rendered video if preferred
}

const ResearchGapDiagram: React.FC<ResearchGapDiagramProps> = ({ remotionVideoSrc }) => {

  return (
    <Box
      sx={{
        height: { xs: 200, sm: 300, md: 400 },
        bgcolor: 'var(--bg-elevated)',
        borderRadius: 'var(--radius-md)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        position: 'relative',
        boxShadow: 'var(--shadow-lg)',
        my: 4,
      }}
    >
      {remotionVideoSrc ? (
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
      ) : (
        <Box sx={{ width: '100%', height: '100%' }}>
            <Player
                component={ResearchGapComposition}
                durationInFrames={200}
                fps={30}
                compositionWidth={800}
                compositionHeight={400}
                style={{
                    width: '100%',
                    height: '100%',
                }}
                inputProps={{}}
                loop
                autoPlay
                controls={false}
            />
            {/* Overlay description if needed, or included in animation */}
             <Typography
                variant="caption"
                sx={{ 
                    position: 'absolute', 
                    bottom: 8, 
                    left: 16, 
                    color: 'var(--text-muted)',
                    bgcolor: 'rgba(0,0,0,0.5)',
                    px: 1,
                    borderRadius: 1,
                    zIndex: 20
                }}
              >
                Illustrates how Scholar Agent pinpoints research opportunities.
              </Typography>
        </Box>
      )}
    </Box>
  );
};

export default ResearchGapDiagram;
