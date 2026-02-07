import React from 'react';
import { motion } from 'framer-motion';
import { ValueCard } from './ValueCard';
import {
  Speed as SpeedIcon,
  Search as SearchIcon,
  Group as GroupIcon,
  Visibility as VisibilityIcon,
} from '@mui/icons-material';

interface ValueCardData {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const valueCards: ValueCardData[] = [
  {
    icon: <SpeedIcon sx={{ fontSize: 32 }} />,
    title: 'Time Savings',
    description: 'Reduce weeks of manual research to minutes. Our multi-agent system works in parallel to analyze papers, identify gaps, and synthesize insights faster than ever before.',
  },
  {
    icon: <SearchIcon sx={{ fontSize: 32 }} />,
    title: 'Research Gap Identification',
    description: 'Discover untapped research opportunities with AI-powered gap analysis. Our system identifies areas where your field needs more investigation.',
  },
  {
    icon: <GroupIcon sx={{ fontSize: 32 }} />,
    title: 'Multi-Agent Intelligence',
    description: 'Leverage specialized AI agents working together: planners, retrievers, analyzers, and synthesizers collaborate to deliver comprehensive research insights.',
  },
  {
    icon: <VisibilityIcon sx={{ fontSize: 32 }} />,
    title: 'Real-time Transparency',
    description: 'See exactly what each agent is doing in real-time. Every step of the research process is visible, traceable, and understandable.',
  },
];

export const ValuePropositionGrid: React.FC = () => {
  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-[var(--bg-surface)]">
      <div className="container mx-auto max-w-7xl">
        {/* Section Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2
            className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-6 text-[var(--text-primary)]"
            style={{ fontFamily: 'var(--font-primary)' }}
          >
            Why Scholar Agent?
          </h2>
          <p
            className="text-xl text-[var(--text-secondary)] max-w-3xl mx-auto"
            style={{ fontFamily: 'var(--font-content)' }}
          >
            Experience the future of academic research with AI-powered agents that work tirelessly to accelerate your discoveries.
          </p>
        </motion.div>

        {/* Value Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
          {valueCards.map((card, index) => (
            <ValueCard
              key={index}
              icon={card.icon}
              title={card.title}
              description={card.description}
              delay={index * 0.1}
            />
          ))}
        </div>
      </div>
    </section>
  );
};
