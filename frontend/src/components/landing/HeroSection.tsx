import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { AnimatedWorkflow } from "./AnimatedWorkflow";

interface HeroSectionProps {
  headline?: string;
  subheadline?: string;
  primaryCTA?: string;
  secondaryCTA?: string;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  headline = "Transform Weeks of Research Into Minutes",
  subheadline = "Scholar Agent uses multi-agent AI to identify research gaps, analyze papers, and synthesize insights—all with complete transparency.",
  primaryCTA = "Start Your First Research",
  secondaryCTA = "Learn More",
}) => {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: "easeOut" as const,
      },
    },
  };

  return (
    <section
      className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-[var(--bg-page)] via-[var(--bg-surface)] to-[var(--bg-page)]"
      aria-label="Hero section"
    >
      {/* Background decorative elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-[var(--accent-secondary)] rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob"></div>
        <div className="absolute top-40 right-10 w-72 h-72 bg-[var(--accent-success)] rounded-full mix-blend-screen filter blur-3xl opacity-25 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-1/2 w-72 h-72 bg-[var(--accent-primary)] rounded-full mix-blend-screen filter blur-3xl opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Text Content */}
          <motion.div
            className="text-center lg:text-left mt-12 sm:mt-0"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.h1
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-extrabold mb-6 leading-tight tracking-tight"
              variants={itemVariants}
              style={{
                fontFamily: "var(--font-primary)",
                background:
                  "linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
              }}
            >
              {headline}
            </motion.h1>

            <motion.p
              className="text-lg sm:text-xl md:text-2xl mb-8 text-[var(--text-secondary)] leading-relaxed max-w-2xl mx-auto lg:mx-0"
              variants={itemVariants}
              style={{ fontFamily: "var(--font-content)" }}
            >
              {subheadline}
            </motion.p>

            <motion.div
              className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start"
              variants={itemVariants}
            >
              <motion.button
                onClick={() => navigate("/register")}
                className="px-8 py-4 rounded-[var(--radius-md)] font-semibold text-lg text-[var(--bg-page)] shadow-lg hover:shadow-xl transition-all duration-[var(--transition-standard)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)] focus:ring-offset-2 focus:ring-offset-[var(--bg-page)]"
                style={{
                  background:
                    "linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)",
                  fontFamily: "var(--font-primary)",
                }}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.98 }}
                aria-label="Start your first research"
              >
                {primaryCTA}
              </motion.button>

              <motion.button
                onClick={() => navigate("/how-it-works")}
                className="px-8 py-4 rounded-[var(--radius-md)] font-semibold text-lg border-2 border-[var(--accent-secondary)] text-[var(--accent-secondary)] bg-transparent hover:bg-[color-mix(in_oklab,var(--accent-secondary)_8%,transparent)] transition-all duration-[var(--transition-standard)] focus:outline-none focus:ring-2 focus:ring-[var(--accent-secondary)] focus:ring-offset-2 focus:ring-offset-[var(--bg-page)]"
                style={{ fontFamily: "var(--font-primary)" }}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.98 }}
                aria-label="Learn more about Scholar Agent"
              >
                {secondaryCTA}
              </motion.button>
            </motion.div>
          </motion.div>

          {/* 3D Visualization */}
          <motion.div
            className="relative h-[400px] sm:h-[500px] lg:h-[600px] w-full"
            variants={itemVariants}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
          >
            <AnimatedWorkflow />
          </motion.div>
        </div>
      </div>

      <style>{`
        @keyframes blob {
          0%, 100% {
            transform: translate(0, 0) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </section>
  );
};
