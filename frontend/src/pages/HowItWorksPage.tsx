import React from "react";
import { Box, Typography, Container, Button } from "@mui/material";
import { useNavigate } from "react-router-dom";
import AgentJourney from "../components/how-it-works/AgentJourney";
import ResearchGapDiagram from "../components/how-it-works/ResearchGapDiagram";
import ExampleOutputPreview from "../components/how-it-works/ExampleOutputPreview";
import ComparisonTable from "../components/how-it-works/ComparisonTable";
import FAQAccordion from "../components/how-it-works/FAQAccordion";



const faqs = [
  {
    question: "How accurate are the generated literature reviews?",
    answer:
      "Scholar Agent leverages state-of-the-art AI models and a multi-agent validation system to ensure high accuracy and reliability. Our Quality Checker agent specifically refines and validates findings before synthesis.",
  },
  {
    question: "What sources does Scholar Agent use for research?",
    answer:
      "The Retriever Agent accesses and processes vast academic databases, journals, and reputable online sources to ensure comprehensive coverage and up-to-date information.",
  },
  {
    question: "Can I customize the output of the literature reviews?",
    answer:
      "Yes, Scholar Agent offers various customization options. You can specify parameters such as scope, depth, style, and focus areas to tailor the literature review to your specific needs.",
  },
  {
    question: "Is my research data kept private and secure?",
    answer:
      "Absolutely. We prioritize user privacy and data security. All research data and personal information are encrypted and handled in strict compliance with privacy regulations.",
  },
];

const HowItWorksPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100%',
        backgroundColor: 'var(--bg-page)',
      }}
    >
      <Container
        maxWidth="lg"
        sx={{
          py: 8,
          color: 'var(--text-primary)',
        }}
      >
      <Box sx={{ textAlign: "center", mb: 6 }}>
        <Typography
          variant="h2"
          component="h1"
          gutterBottom
          sx={{ color: "var(--text-primary)" }}
        >
          How Scholar Agent Works
        </Typography>
        <Typography variant="h5" sx={{ mb: 4, color: "var(--text-secondary)" }}>
          Understanding the Multi-Agent System Behind Your Research
        </Typography>
        <Button
          variant="contained"
          onClick={() => navigate("/")}
          sx={{
            mr: 2,
            backgroundColor: "var(--accent-primary)",
            color: "white",
            "&:hover": {
              backgroundColor: "var(--accent-secondary)",
            },
          }}
        >
          Back to Landing
        </Button>
        <Button
          variant="outlined"
          onClick={() => navigate("/register")}
          sx={{
            borderColor: "var(--accent-primary)",
            color: "var(--accent-primary)",
            "&:hover": {
              borderColor: "var(--accent-secondary)",
              color: "var(--accent-secondary)",
            },
          }}
        >
          Sign Up Now
        </Button>
      </Box>

      {/* Multi-Agent Pipeline Journey Section */}
      <Box sx={{ mb: 8 }}>
        <Typography
          variant="h3"
          gutterBottom
          sx={{ color: "var(--text-primary)", textAlign: 'center', mb: 4 }}
        >
          The Multi-Agent Pipeline
        </Typography>
        <Typography
          variant="body1"
          sx={{ color: "var(--text-secondary)", textAlign: 'center', mb: 6, maxWidth: 600, mx: 'auto' }}
        >
          Watch how our intelligent agents collaborate seamlessly to transform your research queries into comprehensive literature reviews.
        </Typography>
        <AgentJourney />
      </Box>

      {/* Research Gap Identification Section */}
      <Box sx={{ mb: 8 }}>
        <Typography
          variant="h3"
          gutterBottom
          sx={{ color: "var(--text-primary)" }}
        >
          Identifying Research Gaps
        </Typography>
        <ResearchGapDiagram />
        <Typography
          variant="body1"
          sx={{ mt: 2, color: "var(--text-secondary)" }}
        >
          Scholar Agent employs sophisticated algorithms to identify novel
          research opportunities and unaddressed questions within the existing
          literature, guiding your work towards impactful contributions.
        </Typography>
      </Box>

      {/* Example Output Section */}
      <Box sx={{ mb: 8 }}>
        <Typography
          variant="h3"
          gutterBottom
          sx={{ color: "var(--text-primary)" }}
        >
          Example Output
        </Typography>
        <ExampleOutputPreview
          title="Sample Literature Review Excerpt"
          excerpt="This excerpt demonstrates Scholar Agent's ability to synthesize complex information into a coherent and insightful literature review, highlighting key themes and findings from diverse sources."
          fullExampleLink="/example-full-review"
        />
      </Box>

      {/* Comparison Table Section */}
      <Box sx={{ mb: 8 }}>
        <Typography
          variant="h3"
          gutterBottom
          sx={{ color: "var(--text-primary)" }}
        >
          Manual vs. Scholar Agent
        </Typography>
        <ComparisonTable />
      </Box>

      {/* FAQ Section */}
      <Box sx={{ mb: 8 }}>
        <Typography
          variant="h3"
          gutterBottom
          sx={{ color: "var(--text-primary)" }}
        >
          Frequently Asked Questions
        </Typography>
        <FAQAccordion faqs={faqs} />
      </Box>
    </Container>
    </Box>
  );
};

export default HowItWorksPage;
