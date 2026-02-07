import React, { useMemo } from 'react';
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  random,
} from 'remotion';
import { Box } from '@mui/material';
import { Lightbulb, Search } from '@mui/icons-material';

// Design System Colors - Hardcoded for Remotion Player isolation
const COLORS = {
  Paper: '#A1A1AA', // --color-obsidian-300
  Connection: '#D4D4D8', // --color-obsidian-200
  Scanner: '#00F5C8', // --color-aurora-500
  Gap: '#FFB900', // --color-insight-500
  Background: '#27272F', // --bg-elevated / --color-obsidian-700
};

const DOT_COUNT = 30;
const SEED = 1234;

export const ResearchGapComposition: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Generate static positions for "papers" (dots)
  // We want to avoid the center area to create a "gap"
  const papers = useMemo(() => {
    return new Array(DOT_COUNT).fill(0).map((_, i) => {
      const angle = random(SEED + i) * Math.PI * 2;
      // Push dots away from center (gap radius approx 100px)
      const radius = 100 + random(SEED + i + 1) * 200; 
      const x = Math.cos(angle) * radius;
      const y = Math.sin(angle) * radius;
      return { x, y, delay: i * 2 };
    });
  }, []);

  // Scanning effect opacity
  const scanOpacity = interpolate(
    frame,
    [30, 90], // Scan happens from 1s to 3s
    [0, 1],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  
  const scanRotation = interpolate(
      frame,
      [30, 150],
      [0, 360 * 2], // 2 full rotations
      { extrapolateLeft: 'clamp' }
  );

  // Gap discovery (The "Aha!" moment)
  const discoveryScale = spring({
    frame: frame - 120, // Starts after scanning
    fps,
    config: { damping: 10, stiffness: 100 },
  });

  const discoveryOpacity = interpolate(
     frame,
     [120, 140],
     [0, 1],
     { extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}
  );

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.Background,
        justifyContent: 'center',
        alignItems: 'center',
        overflow: 'hidden',
      }}
    >
      {/* 1. Existing Literature (Dots appearing) */}
      <Box sx={{ position: 'relative', width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          {papers.map((paper, i) => {
             const dotScale = spring({
                 frame: frame - paper.delay,
                 fps, 
                 config: { damping: 10 }
             });
             
             // Draw connections between close nodes (simplified visually)
             // We can just draw lines to a few neighbors if we wanted, but let's keep it simple: just dots for now
             
             return (
                 <Box
                    key={i}
                    sx={{
                        position: 'absolute',
                        left: '50%',
                        top: '50%',
                        width: 8,
                        height: 8,
                        marginLeft: '-4px', // Centering correction
                        marginTop: '-4px', // Centering correction
                        borderRadius: '50%',
                        bgcolor: COLORS.Paper,
                        transform: `translate(${paper.x}px, ${paper.y}px) scale(${dotScale})`,
                        opacity: 0.6
                    }}
                 />
             );
          })}
          
          {/* 2. Scanning Radar Effect */}
          <Box
             sx={{
                 position: 'absolute',
                 left: '50%',
                 top: '50%',
                 marginLeft: '-200px', // Half of width
                 marginTop: '-200px', // Half of height
                 width: 400,
                 height: 400,
                 borderRadius: '50%',
                 border: `2px solid ${COLORS.Scanner}`,
                 borderTopColor: 'transparent',
                 borderLeftColor: 'transparent',
                 opacity: scanOpacity > 0 ? 0.3 : 0,
                 transform: `rotate(${scanRotation}deg)`,
                 boxShadow: `inset 0 0 50px ${COLORS.Scanner}`
             }}
          />
          
           {/* Scanning Icon/Text */}
           <Box
              sx={{
                  position: 'absolute',
                  top: '10%',
                  left: 0,
                  width: '100%',
                  opacity: interpolate(frame, [30, 60, 110, 130], [0, 1, 1, 0]),
                  display: 'flex', 
                  flexDirection: 'column', 
                  alignItems: 'center',
                  gap: 1
              }}
           >
               <Search sx={{ color: COLORS.Scanner, fontSize: 32 }} />
               <Box sx={{ color: COLORS.Scanner, typography: 'caption', fontWeight: 'bold' }}>SCANNING LITERATURE</Box>
           </Box>

          {/* 3. The Gap Identified */}
          <Box
             sx={{
                 position: 'absolute',
                 left: '50%',
                 top: '50%',
                 transform: `translate(-50%, -50%) scale(${discoveryScale})`,
                 opacity: discoveryOpacity,
                 display: 'flex',
                 flexDirection: 'column',
                 alignItems: 'center',
                 zIndex: 10
             }}
          >
              <Box
                sx={{
                    position: 'relative',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                }}
              >
                  {/* Glowing background for the gap */}
                  <Box 
                    sx={{ 
                        position: 'absolute', 
                        width: 60, 
                        height: 60, 
                        borderRadius: '50%', 
                        bgcolor: COLORS.Gap, 
                        filter: 'blur(20px)',
                        opacity: 0.5
                    }} 
                  />
                  <Lightbulb sx={{ fontSize: 60, color: COLORS.Gap }} />
              </Box>
              
              <Box 
                sx={{ 
                    mt: 2, 
                    color: COLORS.Gap, 
                    fontWeight: 'bold', 
                    typography: 'subtitle1',
                    textShadow: `0 0 10px ${COLORS.Gap}`,
                    textAlign: 'center',
                    whiteSpace: 'nowrap'
                }}
            >
                  RESEARCH GAP IDENTIFIED
              </Box>
          </Box>
      </Box>
    </AbsoluteFill>
  );
};
