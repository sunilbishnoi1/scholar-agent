import React, { useState } from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

interface FAQItem {
  question: string;
  answer: string;
}

interface FAQAccordionProps {
  faqs: FAQItem[];
}

const FAQAccordion: React.FC<FAQAccordionProps> = ({ faqs }) => {
  const [expanded, setExpanded] = useState<string | false>(false);

  const handleChange = (panel: string) => (_event: React.SyntheticEvent, newExpanded: boolean) => {
    setExpanded(newExpanded ? panel : false);
  };

  return (
    <Box sx={{ my: 4 }}>
      {faqs.map((faq, index) => (
        <Accordion
          key={`faq-${index}`}
          expanded={expanded === `panel${index}`}
          onChange={handleChange(`panel${index}`)}
          sx={{
            mb: 1,
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            backgroundColor: 'var(--bg-surface)',
            '&.Mui-expanded': {
              margin: '8px 0',
              boxShadow: 'var(--shadow-lg)',
            },
            '&:before': {
              display: 'none', // Remove default Accordion border
            },
          }}
        >
          <AccordionSummary
            expandIcon={<ExpandMoreIcon sx={{ color: 'var(--accent-primary)' }} />}
            aria-controls={`panel${index}-content`}
            id={`panel${index}-header`}
            sx={{
              backgroundColor: 'var(--bg-elevated)',
              borderRadius: expanded === `panel${index}` ? 'var(--radius-md) var(--radius-md) 0 0' : 'var(--radius-md)',
              '& .MuiAccordionSummary-content': {
                my: 2,
              },
            }}
          >
            <Typography variant="h6" component="span" sx={{ color: 'var(--text-primary)' }}>
              {faq.question}
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 3, backgroundColor: 'var(--bg-surface)' }}>
            <Typography variant="body1" sx={{ color: 'var(--text-secondary)' }}>
              {faq.answer}
            </Typography>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default FAQAccordion;
