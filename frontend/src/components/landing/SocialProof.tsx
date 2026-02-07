import React from 'react';
import { motion } from 'framer-motion';
import { Star as StarIcon } from '@mui/icons-material';

interface Testimonial {
  name: string;
  role: string;
  affiliation: string;
  content: string;
  rating: number;
}

interface Stat {
  value: string;
  label: string;
}

const testimonials: Testimonial[] = [
  {
    name: 'Dr. Sarah Chen',
    role: 'Research Professor',
    affiliation: 'MIT Computer Science',
    content: 'Scholar Agent transformed how I approach literature reviews. What used to take weeks now takes hours, and the gap analysis is incredibly insightful.',
    rating: 5,
  },
  {
    name: 'Prof. James Rodriguez',
    role: 'Department Head',
    affiliation: 'Stanford AI Lab',
    content: 'The transparency of the multi-agent system is remarkable. I can see exactly how each agent contributes to the research process, which builds trust.',
    rating: 5,
  },
  {
    name: 'Dr. Emily Watson',
    role: 'Postdoctoral Researcher',
    affiliation: 'Harvard Medical School',
    content: 'As someone new to a research field, Scholar Agent helped me quickly identify key papers and understand the research landscape. It\'s like having a team of expert researchers working for you.',
    rating: 5,
  },
];

const stats: Stat[] = [
  { value: '10,000+', label: 'Research Papers Analyzed' },
  { value: '500+', label: 'Active Researchers' },
  { value: '95%', label: 'Time Saved' },
  { value: '4.9/5', label: 'User Rating' },
];

export const SocialProof: React.FC = () => {
  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-[var(--bg-surface)] to-[var(--bg-page)]">
      <div className="container mx-auto max-w-7xl">
        {/* Stats Section */}
        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-20"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              className="text-center"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <div
                className="text-4xl md:text-5xl font-extrabold mb-2"
                style={{
                  background: 'linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  fontFamily: 'var(--font-primary)',
                }}
              >
                {stat.value}
              </div>
              <p
                className="text-[var(--text-secondary)] text-sm md:text-base"
                style={{ fontFamily: 'var(--font-primary)' }}
              >
                {stat.label}
              </p>
            </motion.div>
          ))}
        </motion.div>

        {/* Testimonials Section */}
        <div>
          <motion.h2
            className="text-4xl sm:text-5xl font-extrabold text-center mb-16 text-[var(--text-primary)]"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            style={{ fontFamily: 'var(--font-primary)' }}
          >
            Trusted by Researchers Worldwide
          </motion.h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                className="p-8 rounded-[var(--radius-lg)] bg-[var(--bg-elevated)] border border-[var(--border-subtle)] shadow-md hover:shadow-xl transition-shadow duration-[var(--transition-standard)]"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -4 }}
              >
                {/* Rating */}
                <div className="flex gap-1 mb-4">
                  {Array.from({ length: testimonial.rating }).map((_, i) => (
                    <StarIcon
                      key={i}
                      sx={{ fontSize: 20, color: '#FFB900' }}
                    />
                  ))}
                </div>

                {/* Content */}
                <p
                  className="text-[var(--text-secondary)] mb-6 leading-relaxed italic"
                  style={{ fontFamily: 'var(--font-content)' }}
                >
                  "{testimonial.content}"
                </p>

                {/* Author */}
                <div>
                  <p
                    className="font-semibold text-[var(--text-primary)]"
                    style={{ fontFamily: 'var(--font-primary)' }}
                  >
                    {testimonial.name}
                  </p>
                  <p
                    className="text-sm text-[var(--text-secondary)]"
                    style={{ fontFamily: 'var(--font-primary)' }}
                  >
                    {testimonial.role}
                  </p>
                  <p
                    className="text-sm text-[var(--text-muted)]"
                    style={{ fontFamily: 'var(--font-primary)' }}
                  >
                    {testimonial.affiliation}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
