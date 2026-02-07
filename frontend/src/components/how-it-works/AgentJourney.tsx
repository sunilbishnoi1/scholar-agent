import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Box } from '@mui/material';
import AgentJourneyCard from './AgentJourneyCard';
import { type AgentData, agentColors } from '../../types/agent';
import './AgentJourney.css';

// Agents data
const agentsData: AgentData[] = [
  {
    name: 'Planner Agent',
    role: 'Strategizes Research Approach',
    description:
      'Formulates a comprehensive strategy, breaking down complex research questions into manageable sub-tasks. Lays the groundwork for efficient and targeted literature review.',
  },
  {
    name: 'Retriever Agent',
    role: 'Discovers Relevant Papers',
    description:
      'Efficiently identifies and retrieves the most pertinent academic papers from vast databases, ensuring a broad and relevant knowledge base.',
  },
  {
    name: 'Analyzer Agent',
    role: 'Performs Deep Paper Analysis',
    description:
      'Conducts in-depth analysis of retrieved documents, extracting key findings, methodologies, and arguments from the research landscape.',
  },
  {
    name: 'Quality Checker Agent',
    role: 'Validates and Refines Findings',
    description:
      'Ensures accuracy, coherence, and logical consistency of analyzed information, flagging discrepancies and suggesting refinements.',
  },
  {
    name: 'Synthesizer Agent',
    role: 'Generates Literature Reviews',
    description:
      'Synthesizes insights from all agents into a cohesive, well-structured, and high-quality literature review tailored to your research.',
  },
];

const AgentJourney: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const anchorRefs = useRef<(HTMLDivElement | null)[]>([]);
  
  const [svgPath, setSvgPath] = useState<string>('');
  const [pathLength, setPathLength] = useState<number>(0);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  // Path SVG element ref to measure length
  const pathRef = useRef<SVGPathElement>(null);

  // Register anchor refs
  const setAnchorRef = useCallback((el: HTMLDivElement | null, index: number) => {
    anchorRefs.current[index] = el;
  }, []);

  // Calculate SVG Path
  const calculatePath = useCallback(() => {
    if (!containerRef.current || anchorRefs.current.length === 0) return;
    
    // Check if we have valid refs
    const validRefs = anchorRefs.current.every(ref => ref?.getBoundingClientRect);
    if (!validRefs) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const points = anchorRefs.current.map(ref => {
      if (!ref) return { x: 0, y: 0 };
      const rect = ref.getBoundingClientRect();
      return {
        x: rect.left - containerRect.left + rect.width / 2,
        y: rect.top - containerRect.top + rect.height / 2
      };
    });

    // Generate Cubic Bezier Path
    let d = `M ${points[0].x} ${points[0].y}`;
    
    for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i];
        const p2 = points[i + 1];
        

        


        // Curve logic:
        // p1 -> cp1 (x + offset? No, y based) -> cp2 -> p2
        // We want S curve. p1 is left, p2 is right.
        // Or p1 needs to curve OUT horizontally first?
        
        // Simple S-curve:
        // C (p1.x) (p1.y + dy/2) (p2.x) (p2.y - dy/2) p2.x p2.y
        // This is vertical S-curve.
        
        // But our points are also horizontally separated significantly.
        // We want to exit p1 horizontally and enter p2 horizontally?
        // Or exit vertically?
        // Let's try vertical exit for "waterfall" feel or horizontal exit for "flow".
        // Given layout "Line starts near card...", it might look better if it flows vertically between layers.
        
        // Let's use standard vertical cubic bezier:
        // cp1 = p1.x, p1.y + (p2.y - p1.y) * 0.5
        // cp2 = p2.x, p2.y - (p2.y - p1.y) * 0.5
        
        const dy = p2.y - p1.y;
        
        const cp1x = p1.x;
        const cp1y = p1.y + dy * 0.5;
        
        const cp2x = p2.x;
        const cp2y = p2.y - dy * 0.5;
        
        d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2.x} ${p2.y}`;
    }
    
    setSvgPath(d);
  }, []);

  // Measure path length when path changes
  useEffect(() => {
      if (pathRef.current) {
          setPathLength(pathRef.current.getTotalLength());
      }
  }, [svgPath]);

  // Handle Resize and Scroll
  useEffect(() => {
    // Initial calculation after a brief delay to ensure layout is settled
    const timer = setTimeout(calculatePath, 100);
    
    window.addEventListener('resize', calculatePath);
    
    return () => {
        clearTimeout(timer);
        window.removeEventListener('resize', calculatePath);
    };
  }, [calculatePath]);

  // Recalculate on scroll (for sticky nav bars adjustment etc, though mostly resize matters)
  // Actually, position of elements inside container doesn't change on scroll RELATIVE to container usually.
  // But we need to track scroll progress.

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const componentHeight = rect.height;

    // Calculate progress
    // We want the line to draw as the center of the viewport hits the sections.
    // Or just simple intersection ratio.
    
    const currentPos = (windowHeight * 0.5) - rect.top; // How far we are into the component
    
    // Clamp 0 to 1
    const progress = Math.max(0, Math.min(1, currentPos / (componentHeight * 0.9))); 
    // 0.9 factor ensures it fills before we completely scroll past
    
    setScrollProgress(progress);

    // Active index based on progress
    const agentIndex = Math.floor(progress * agentsData.length);
    setActiveIndex(Math.min(agentIndex, agentsData.length - 1));
    
  }, []);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll(); // Initial check
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  return (
    <Box
      ref={containerRef}
      className="agent-journey"
      sx={{ position: 'relative', py: 8 }}
    >
      {/* SVG Container */}
      <svg 
        className="agent-journey__svg-container"
        style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 1
        }}
      >
        <defs>
            <linearGradient id="pipelineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
               {agentsData.map((agent, i) => (
                   <stop 
                        key={i} 
                        offset={`${(i / (agentsData.length - 1)) * 100}%`} 
                        stopColor={agentColors[agent.name]} 
                   />
               ))}
            </linearGradient>
            
            {/* Glow filter */}
             <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        
        {/* Background Path (Grey) */}
        <path
            d={svgPath}
            stroke="var(--border-subtle)"
            strokeWidth="3"
            fill="none"
            strokeLinecap="round"
        />
        
        {/* Animated Progress Path (Colored) */}
        <path
            ref={pathRef}
            d={svgPath}
            stroke="url(#pipelineGradient)"
            strokeWidth="4"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={pathLength}
            strokeDashoffset={pathLength - (pathLength * scrollProgress)}
            style={{ transition: 'stroke-dashoffset 0.1s linear' }}
            filter="url(#glow)"
        />
      </svg>

      {/* Cards Container */}
      <div className="agent-journey__cards">
        {agentsData.map((agent, index) => (
          <AgentJourneyCard
            key={agent.name}
            agent={agent}
            index={index}
            isActive={activeIndex !== null && index <= activeIndex}
            connectionRef={(el) => setAnchorRef(el, index)}
          />
        ))}
      </div>
    </Box>
  );
};

export default AgentJourney;
