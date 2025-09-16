import React from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Divider,
  Avatar,
  Chip,
  Grid,
} from '@mui/material';
import { motion, type Variants } from 'framer-motion';
import MapIcon from '@mui/icons-material/Map';
import ArticleIcon from '@mui/icons-material/Article';
import ScienceIcon from '@mui/icons-material/Science';
import EditNoteIcon from '@mui/icons-material/EditNote';
import MailOutlineIcon from '@mui/icons-material/MailOutline';
import ShutterSpeedIcon from '@mui/icons-material/ShutterSpeed';
import FindInPageIcon from '@mui/icons-material/FindInPage';
import arxivLogo from '../assets/arxiv-logo.png';
import semanticLogo from '../assets/semantic-logo.png';

// Motion helpers (typed correctly)
const containerVariant: Variants = {
  hidden: {},
  show: {
    transition: {
      staggerChildren: 0.12,
    },
  },
};

const itemVariant: Variants = {
  hidden: { opacity: 0, y: 18 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: 'easeOut' },
  },
};

const AnimatedSection: React.FC<{ children: React.ReactNode; sx?: any }> = ({ children, sx }) => {
  return (
    <motion.div
      variants={containerVariant}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.18 }}
      style={{ width: '100%' }}
    >
      <Box sx={sx}>{children}</Box>
    </motion.div>
  );
};

const StepCard: React.FC<{ icon: React.ReactNode; title: string; desc: string; sx?: any }> = ({ icon, title, desc, sx }) => {
  return (
    <motion.div variants={itemVariant} whileHover={{ y: -6, boxShadow: '0 10px 30px rgba(0,0,0,0.08)' }}>
      <Box
        sx={{
          textAlign: 'center',
          px: { xs: 2, md: 1 },
          py: { xs: 3, md: 4 },
          minHeight: 180,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'flex-start',
          ...sx,
        }}
      >
        <Avatar sx={{ bgcolor: 'primary.main', width: { xs: 52, md: 64 }, height: { xs: 52, md: 64 }, mb: 2 }}>
          {icon}
        </Avatar>
        <Typography variant="h6" fontWeight={700} sx={{ mb: 1 }}>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 320 }}>
          {desc}
        </Typography>
      </Box>
    </motion.div>
  );
};

const KnowPage: React.FC = () => {
  // const theme = useTheme();
  // const isSm = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Container maxWidth="lg" sx={{ py: { xs: 12, md: 12 }, overflow: 'hidden' }}>
      {/* HERO */}
      <Box sx={{ textAlign: 'center', mb: { xs: 3, md: 4 }, px: { xs: 2, md: 0 } }}>
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Typography
            component="h1"
            sx={{
              fontWeight: 800,
              fontSize: { xs: '1.6rem', md: '2.4rem' },
              lineHeight: 1.05,
              background: 'linear-gradient(90deg,#0ea5a4,#2563eb)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Your Intelligent Research Partner
          </Typography>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.12 }}
        >
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{ mt: 2, maxWidth: 780, mx: 'auto' }}
          >
            Discover how our multi-agent system transforms your complex research questions into synthesized, actionable insights in a fraction of the time.
          </Typography>
        </motion.div>
      </Box>

      {/* HOW IT WORKS */}
      <AnimatedSection sx={{ mb: { xs: 2, md: 8 } }}>
  <Typography
    variant="h5"
    component="h2"
    sx={{ textAlign: 'center', fontWeight: 800, mb: { xs: 2, md: 4 } }}
  >
    From Hours to Minutes: A 4-Step Journey
  </Typography>

  <Grid container spacing={{ xs: 3, md: 4 }} justifyContent="center">
    {[
      {
        icon: <EditNoteIcon fontSize="large" />,
        title: '1. You Ask',
        desc: 'Simply provide your research question and title. Our system gets to work immediately.',
      },
      {
        icon: <MapIcon fontSize="large" />,
        title: '2. We Plan & Search',
        desc: 'The Planner Agent creates a strategy and scours leading academic databases for relevant papers.',
      },
      {
        icon: <ScienceIcon fontSize="large" />,
        title: '3. Agents Analyze',
        desc: 'Each paper is meticulously analyzed for findings, methods, and limitations by our specialist agent.',
      },
      {
        icon: <ArticleIcon fontSize="large" />,
        title: '4. You Get Insights',
        desc: 'The Synthesizer Agent drafts a complete literature review and emails it directly to you.',
      },
    ].map((step, idx) => (
      <Grid {...({ item: true } as any)} xs={12} sm={6} md={3} key={idx}>
        <StepCard
          icon={step.icon}
          title={step.title}
          desc={step.desc}
          sx={{
            position: 'relative',
            overflow: 'visible',
            filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
            mt: 1.5,
            minWidth: 220,
            borderRadius: 4,
            backgroundColor: 'rgba(255, 255, 255, 0.5)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            
          }}
        />
      </Grid>
    ))}
  </Grid>
</AnimatedSection>


      <Divider sx={{ my: { xs: 5, md: 8 } }} />

      {/* AGENTS */}
      <Box sx={{ mb: { xs: 6, md: 10 } }}>
        <Typography
          variant="h5"
          component="h2"
          sx={{ textAlign: 'center', fontWeight: 800, mb: 6 }}
        >
          Meet Your AI Research Team
        </Typography>

        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' }, // column on mobile, row on large screens
            gap: { xs: 2, md: 4 },
            alignItems: 'stretch',
            justifyContent: 'center',
          }}
        >
          {[
            {
              icon: <MapIcon sx={{ fontSize: 40 }} color="primary" />,
              title: 'The Research Planner',
              desc:
                'This agent acts as your strategist. It deconstructs your research question to generate precise keywords and subtopics, ensuring the search is both comprehensive and focused.',
            },
            {
              icon: <FindInPageIcon sx={{ fontSize: 40 }} color="primary" />,
              title: 'The Paper Analyzer',
              desc:
                'The detail-oriented expert. It reads and dissects every retrieved paper, extracting key findings, methodologies, contributions, and—crucially—their limitations.',
            },
            {
              icon: <ArticleIcon sx={{ fontSize: 40 }} color="primary" />,
              title: 'The Synthesis Executor',
              desc:
                'Your personal academic writer. It weaves together the analyzed data into a coherent literature review, critically comparing findings and highlighting the research gaps you need to find.',
            },
          ].map((a, idx) => (
            <Box
              key={idx}
              sx={{
                width: { xs: '100%', md: '33.333%' }, // full width on mobile, 1/3 on md+
                display: 'flex', // keep Paper stretched to same height when in row
              }}
            >
              <motion.div variants={itemVariant} whileHover={{ y: -6 }} style={{ width: '100%' }}>
                <Paper
                  elevation={0}
                  sx={{
                    p: { xs: 3, md: 4 },
                    borderRadius: 2,
                    height: '100%',
                    bgcolor: 'transparent',
                    border: '1px solid',
                    borderColor: 'divider',
                    transition: 'transform 200ms ease',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {a.icon}
                    <Typography variant="h6" sx={{ fontWeight: 700, ml: 1 }}>
                      {a.title}
                    </Typography>
                  </Box>

                  <Typography variant="body2" color="text.secondary">
                    {a.desc}
                  </Typography>
                </Paper>
              </motion.div>
            </Box>
          ))}
        </Box>
      </Box>


      <Divider sx={{ my: { xs: 5, md: 8 } }} />

      {/* FEATURES & BENEFITS */}
      <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: { xs: 4, md: 6 }, alignItems: 'stretch' }}>
        {/* LEFT COLUMN */}
        <Box sx={{ width: { xs: '100%', md: '50%' }, display: 'flex', flexDirection: 'column', gap: 3 }}>
          <motion.div variants={itemVariant} whileHover={{ y: -6 }} style={{ width: '100%' }}>
            <Box sx={{ height: '100%', p: { xs: 0, md: 0 } }}>
              <Chip icon={<ShutterSpeedIcon />} label="TIME SAVING" color="primary" sx={{ mb: 2 }} />

              <Typography variant="h6" sx={{ fontWeight: 800, mb: 1.5 }}>
                Save Over 40 Hours Per Review
              </Typography>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Manually analyzing up to 50 research papers can take a full week of work. Scholar Agent automates the entire process of searching, reading, analyzing, and synthesizing, freeing you to focus on innovation.
              </Typography>

              <Typography
                variant="h3"
                sx={{
                  fontWeight: 900,
                  fontSize: { xs: '2.6rem', md: '3.6rem' },
                  lineHeight: 1,
                  background: 'linear-gradient(90deg,#06b6d4,#0ea5a4)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 1,
                }}
              >
                95%
              </Typography>

              <Typography variant="subtitle2" color="text.secondary">
                Reduction in Manual Effort
              </Typography>
            </Box>
          </motion.div>

          <motion.div variants={itemVariant} whileHover={{ y: -6 }}>
            <Box>
              <Chip icon={<ArticleIcon />} label="DATA SOURCES" color="primary" sx={{ mb: 2 }} />
              <Typography variant="h6" sx={{ fontWeight: 800, mb: 1.5 }}>
                Powered by Trusted Sources
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                We retrieve papers from leading open-access academic databases, ensuring the foundation of your research is solid and credible.
              </Typography>

              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Box component="img" src={arxivLogo} alt="arXiv Logo" sx={{ height: 28, objectFit: 'contain' }} />
                <Box component="img" src={semanticLogo} alt="Semantic Scholar Logo" sx={{ height: 28, objectFit: 'contain' }} />
              </Box>
            </Box>
          </motion.div>
        </Box>

  {/* RIGHT COLUMN (stacked items) */}
  <Box sx={{ width: { xs: '100%', md: '50%' }, display: 'flex', flexDirection: 'column', gap: 3 }}>
    <motion.div variants={itemVariant} whileHover={{ y: -6 }}>
      <Box>
        <Chip icon={<FindInPageIcon />} label="RESEARCH GAPS" color="primary" sx={{ mb: 2 }} />
        <Typography variant="h6" sx={{ fontWeight: 800, mb: 1.5 }}>
          Discover Research Gaps with Ease
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Our Synthesizer agent is specifically designed to compare and contrast findings from various papers. By identifying contradictions and stated limitations in existing studies, it shines a spotlight on underexplored areas, helping you position your research for maximum impact.
        </Typography>
      </Box>
    </motion.div>

    

    <motion.div variants={itemVariant} whileHover={{ y: -6 }}>
      <Paper
        elevation={0}
        sx={{
          p: 2.5,
          borderRadius: 2,
          background: 'rgba(240,245,255,0.7)',
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
          <MailOutlineIcon color="primary" sx={{ mr: 1.5 }} />
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
            Your Report, Delivered
          </Typography>
        </Box>

        <Typography variant="body2" color="text.secondary">
          No need to wait around or keep checking the page. Once your research project is complete, the full, synthesized report is automatically sent to your registered email address, ready for you to review and use.
        </Typography>
      </Paper>
    </motion.div>
  </Box>
</Box>


      <Divider sx={{ my: { xs: 6, md: 10 } }} />
    </Container>
  );
};

export default KnowPage;
