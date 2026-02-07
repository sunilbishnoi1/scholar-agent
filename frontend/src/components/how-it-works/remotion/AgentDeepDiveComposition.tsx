import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Box } from "@mui/material";
import {
  AccountTree,
  Search,
  CheckCircle,
  AutoStories,
  Article,
  FindInPage,
} from "@mui/icons-material";

// Colors from AgentPipelineVisualization.tsx
const COLORS = {
  Planner: "#00F5C8",
  Retriever: "#00B894",
  Analyzer: "#FFB900",
  QualityChecker: "#A1A1AA",
  Synthesizer: "#00B88D",
};

interface AgentDeepDiveCompositionProps {
  agentName: string;
}

const PlannerAnimation = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const nodes = [
    { x: 0, y: -50, delay: 0 },
    { x: -60, y: 40, delay: 10 },
    { x: 60, y: 40, delay: 20 },
    { x: 0, y: 100, delay: 30 },
  ];

  return (
    <Box sx={{ position: "relative", width: "100%", height: "100%", display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      {nodes.map((node, i) => {
        const scale = spring({
          frame: frame - node.delay,
          fps,
          config: { damping: 10 },
        });
        
        // Connections
        const connectionOpacity = interpolate(frame - node.delay - 10, [0, 10], [0, 1], { extrapolateRight: 'clamp'});

        return (
          <React.Fragment key={i}>
             {i > 0 && (
                <Box
                    sx={{
                        position: 'absolute',
                        left: '50%',
                        top: '50%',
                        width: 2,
                        height: 0, // Simplified connection logic for demo
                        bgcolor: COLORS.Planner,
                        opacity: connectionOpacity,
                        transform: `translate(${node.x/2}px, ${node.y/2}px) rotate(${Math.atan2(node.y, node.x)}rad)`,
                         // This is a simplified visual representation
                    }}
                 />
             )}
            <Box
              sx={{
                position: "absolute",
                transform: `translate(${node.x}px, ${node.y}px) scale(${scale})`,
                bgcolor: COLORS.Planner,
                width: 20,
                height: 20,
                borderRadius: "50%",
                boxShadow: `0 0 10px ${COLORS.Planner}`,
              }}
            />
          </React.Fragment>
        );
      })}
       <Box
          sx={{
            position: 'absolute',
            top: '40%',
            opacity: interpolate(frame, [0, 20], [0, 1]),
            transform: 'translateY(-100px)'
          }}
       >
           <AccountTree sx={{ fontSize: 60, color: COLORS.Planner, opacity: 0.2 }} />
       </Box>
    </Box>
  );
};

const RetrieverAnimation = () => {
  const frame = useCurrentFrame();
  const moveX = interpolate(frame, [0, 60], [-50, 50], {
     extrapolateRight: "clamp",
     easing: (t) => Math.sin(t * Math.PI) // Oscillate
  });
  
  const documents = [-1, 0, 1];

  return (
    <Box sx={{ position: "relative", width: "100%", height: "100%", display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      {documents.map((offset, i) => (
          <Article 
            key={i} 
            sx={{ 
                fontSize: 40, 
                color: 'white', 
                opacity: 0.3,
                position: 'absolute',
                transform: `translateX(${offset * 40}px)`
            }} 
          />
      ))}
      <Box
        sx={{
            // Use moveX here
            position: 'absolute',
            transform: `translateX(${moveX}px) translateY(-20px)`,
        }}
      >
          <Search sx={{ fontSize: 50, color: COLORS.Retriever }} />
      </Box>
    </Box>
  );
};

const AnalyzerAnimation = () => {
    const frame = useCurrentFrame();
    const lines = [0, 1, 2, 3];
    
    return (
        <Box sx={{ position: "relative", width: "100%", height: "100%", display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <Box sx={{ width: 60, height: 80, bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 1, p: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
                {lines.map(i => {
                    const width = interpolate(frame - (i * 10), [0, 10], [0, 100], { extrapolateRight: 'clamp' });
                    return (
                        <Box key={i} sx={{ height: 4, width: `${width}%`, bgcolor: i === 1 || i === 3 ? COLORS.Analyzer : 'rgba(255,255,255,0.3)', borderRadius: 2 }} />
                    )
                })}
            </Box>
            <Box sx={{ position: 'absolute', right: '30%', bottom: '30%' }}>
                <FindInPage sx={{ fontSize: 40, color: COLORS.Analyzer }} />
            </Box>
        </Box>
    );
};

const QualityCheckerAnimation = () => {
    const frame = useCurrentFrame();
    const { fps } = useVideoConfig();
    const scale = spring({ frame: frame - 20, fps, config: { damping: 12 } });
    
    return (
         <Box sx={{ position: "relative", width: "100%", height: "100%", display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
             <Article sx={{ fontSize: 80, color: 'rgba(255,255,255,0.2)' }} />
             <Box sx={{ position: 'absolute', transform: `scale(${scale})` }}>
                 <CheckCircle sx={{ fontSize: 60, color: COLORS.QualityChecker, filter: 'drop-shadow(0 0 10px rgba(0,0,0,0.5))' }} />
             </Box>
         </Box>
    );
};

const SynthesizerAnimation = () => {
    const frame = useCurrentFrame();
    // Particles merging to center
    const particles = [0, 1, 2, 3, 4];
    
    return (
         <Box sx={{ position: "relative", width: "100%", height: "100%", display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
             <AutoStories sx={{ fontSize: 60, color: COLORS.Synthesizer, opacity: interpolate(frame, [30, 50], [0, 1]) }} />
             
             {particles.map((_, i) => {
                 const angle = (i / particles.length) * Math.PI * 2;
                 const distance = interpolate(frame, [0, 30], [80, 0], { extrapolateRight: 'clamp' });
                 const opacity = interpolate(frame, [25, 30], [1, 0], { extrapolateRight: 'clamp' });
                 
                 return (
                     <Box 
                        key={i}
                        sx={{
                            position: 'absolute',
                            width: 8,
                            height: 8,
                            bgcolor: COLORS.Synthesizer,
                            borderRadius: '50%',
                            opacity,
                            transform: `translate(${Math.cos(angle) * distance}px, ${Math.sin(angle) * distance}px)`
                        }}
                     />
                 )
             })}
         </Box>
    );
};

export const AgentDeepDiveComposition: React.FC<AgentDeepDiveCompositionProps> = ({
  agentName,
}) => {
  return (
    <AbsoluteFill
      style={{
        backgroundColor: "var(--bg-elevated)",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      {agentName.includes("Planner") && <PlannerAnimation />}
      {agentName.includes("Retriever") && <RetrieverAnimation />}
      {agentName.includes("Analyzer") && <AnalyzerAnimation />}
      {agentName.includes("Quality") && <QualityCheckerAnimation />}
      {agentName.includes("Synthesizer") && <SynthesizerAnimation />}
    </AbsoluteFill>
  );
};
