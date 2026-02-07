import React from 'react';
import { motion } from 'framer-motion';

interface ValueCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  delay?: number;
}

export const ValueCard: React.FC<ValueCardProps> = ({
  icon,
  title,
  description,
  delay = 0,
}) => {
  return (
    <motion.div
      className="relative p-8 rounded-[var(--radius-lg)] backdrop-blur-xl border transition-all duration-[var(--transition-standard)] cursor-pointer group"
      style={{
        background: 'rgba(9, 9, 11, 0.8)',
        borderColor: 'rgba(255, 255, 255, 0.08)',
        boxShadow: 'var(--shadow-md)',
      }}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, delay, ease: [0.4, 0.0, 0.2, 1] }}
      whileHover={{
        y: -8,
        scale: 1.02,
        boxShadow: 'var(--shadow-xl)',
        transition: { duration: 0.2 },
      }}
    >
      {/* Icon Container */}
      <div className="mb-6 flex items-center justify-center w-16 h-16 rounded-[var(--radius-md)] bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-secondary)] text-[var(--bg-page)] group-hover:scale-110 transition-transform duration-[var(--transition-standard)]">
        {icon}
      </div>

      {/* Content */}
      <h3
        className="text-2xl font-bold mb-4 text-[var(--text-primary)]"
        style={{ fontFamily: 'var(--font-primary)' }}
      >
        {title}
      </h3>
      <p
        className="text-[var(--text-secondary)] leading-relaxed"
        style={{ fontFamily: 'var(--font-content)' }}
      >
        {description}
      </p>

      {/* Hover effect overlay */}
      <div
        className="absolute inset-0 rounded-[var(--radius-lg)] opacity-0 group-hover:opacity-100 transition-opacity duration-[var(--transition-standard)] pointer-events-none"
        style={{
          background:
            'linear-gradient(135deg, rgba(255, 185, 0, 0.10) 0%, rgba(0, 245, 200, 0.10) 100%)',
        }}
      />
    </motion.div>
  );
};
