import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowForward as ArrowForwardIcon } from '@mui/icons-material';

export const QuickStartCTA: React.FC = () => {
  const navigate = useNavigate();

  return (
    <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-[var(--bg-surface)] to-[var(--bg-elevated)] relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-[var(--accent-secondary)]/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-[var(--accent-primary)]/20 rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto max-w-4xl relative z-10">
        <motion.div
          className="text-center"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2
            className="text-4xl sm:text-5xl md:text-6xl font-extrabold mb-6 text-[var(--text-primary)]"
            style={{ fontFamily: 'var(--font-primary)' }}
          >
            Ready to Transform Your Research?
          </h2>
          <p
            className="text-xl sm:text-2xl mb-10 text-[var(--text-secondary)] max-w-2xl mx-auto leading-relaxed"
            style={{ fontFamily: 'var(--font-content)' }}
          >
            Join thousands of researchers who are already accelerating their discoveries with Scholar Agent.
          </p>
          <motion.button
            onClick={() => navigate('/register')}
            className="px-10 py-5 rounded-[var(--radius-md)] font-bold text-xl text-[var(--bg-page)] shadow-2xl hover:shadow-3xl transition-all duration-[var(--transition-standard)] inline-flex items-center gap-3 group"
            style={{
              fontFamily: 'var(--font-primary)',
              background:
                'linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)',
            }}
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.98 }}
          >
            Start Your First Research
            <ArrowForwardIcon className="group-hover:translate-x-1 transition-transform duration-[var(--transition-standard)]" />
          </motion.button>
          <p
            className="mt-6 text-[var(--text-secondary)] text-sm"
            style={{ fontFamily: 'var(--font-primary)' }}
          >
            No credit card required • Free trial available
          </p>
        </motion.div>
      </div>
    </section>
  );
};
