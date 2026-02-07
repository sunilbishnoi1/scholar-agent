import React from 'react';

import { motion } from 'framer-motion';
import { Player } from '@remotion/player';
import { AgentDeepDiveComposition } from './remotion/AgentDeepDiveComposition';

import { type AgentData, agentColors } from '../../types/agent';

interface AgentJourneyCardProps {
  agent: AgentData;
  index: number;
  isActive: boolean;
  connectionRef: (el: HTMLDivElement | null) => void;
}

const AgentJourneyCard: React.FC<AgentJourneyCardProps> = ({
  agent,
  index,
  isActive,
  connectionRef,
}) => {
  const agentColor = agentColors[agent.name] || '#00F5C8';
  const isEven = index % 2 === 0;

  // Calculate node activation based on scroll progress
  // Note: Parent controls the connector line now, but we keep node highlight for local effect
  // We might want to pass 'isNodeActive' from parent later for perfect sync
  
  const cardVariants = {
    hidden: {
      opacity: 0,
    },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.8,
        ease: [0.4, 0, 0.2, 1] as const,
      },
    },
  };

  const animationVariants = {
    hidden: {
      opacity: 0,
    },
    visible: {
      opacity: 1,
      transition: {
        duration: 0.8,
        delay: 0.2,
        ease: [0.4, 0, 0.2, 1] as const,
      },
    },
  };

  const CardContent = (
    <motion.div
      className={`agent-journey__card ${isActive ? 'agent-journey__card--active' : ''}`}
      style={{ borderColor: isActive ? agentColor : undefined, color: agentColor }}
      variants={cardVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-100px' }}
    >
      <div className="agent-journey__card-header">
        <div
          className="agent-journey__card-icon"
          style={{ backgroundColor: agentColor }}
        >
          {agent.name.charAt(0)}
        </div>
        <div>
          <h3 className="agent-journey__card-title" style={{ color: agentColor }}>
            {agent.name}
          </h3>
          <p className="agent-journey__card-role">{agent.role}</p>
        </div>
      </div>
      <p className="agent-journey__card-description">{agent.description}</p>
    </motion.div>
  );

  const Anchor = (
    <div
      ref={connectionRef}
      className={`agent-journey__anchor ${isEven ? 'agent-journey__anchor--right' : 'agent-journey__anchor--left'}`}
      style={{
        position: 'absolute',
        top: '50%',
        width: 12,
        height: 12,
        borderRadius: '50%',
        backgroundColor: isActive ? agentColor : 'var(--border-subtle)',
        transform: 'translateY(-50%)',
        transition: 'background-color 0.3s ease',
        zIndex: 10,
      }}
    />
  );

  const AnimationContent = (
    <motion.div
      className="agent-journey__animation"
      style={{ borderColor: isActive ? `${agentColor}40` : undefined }}
      variants={animationVariants}
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: '-100px' }}
    >
      <Player
        component={AgentDeepDiveComposition}
        inputProps={{ agentName: agent.name }}
        durationInFrames={120}
        compositionWidth={600}
        compositionHeight={400}
        fps={30}
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
        }}
        controls={false}
        autoPlay
        loop
      />
    </motion.div>
  );

  return (
    <div className="agent-journey__card-row">
      {/* Desktop Layout: Alternating left/right */}
      <>
        {/* Left side */}
        <div className="agent-journey__content-left">
          {isEven ? (
            <div style={{ position: 'relative', display: 'inline-block' }}>
              {Anchor}
              {CardContent}
            </div>
          ) : AnimationContent}
        </div>

        {/* Center spacing - preserved for layout but empty now */}
        <div className="agent-journey__spacer" />

        {/* Right side */}
        <div className="agent-journey__content-right">
          {isEven ? AnimationContent : (
            <div style={{ position: 'relative', display: 'inline-block' }}>
              {Anchor}
              {CardContent}
            </div>
          )}
        </div>
      </>
    </div>
  );
};

export default AgentJourneyCard;
