import React from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
} from '@mui/material';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CancelOutlinedIcon from '@mui/icons-material/CancelOutlined';

interface ComparisonRow {
  feature: string;
  manual: string | boolean;
  scholarAgent: string | boolean;
}

const comparisonData: ComparisonRow[] = [
  {
    feature: 'Speed of Research',
    manual: 'Slow, time-consuming',
    scholarAgent: 'Rapid, automated',
  },
  {
    feature: 'Depth of Coverage',
    manual: 'Limited by human capacity',
    scholarAgent: 'Comprehensive global search',
  },
  {
    feature: 'Accuracy & Bias',
    manual: 'Prone to human error & bias',
    scholarAgent: 'AI-driven, validated analysis',
  },
  {
    feature: 'Research Gap Identification',
    manual: 'Challenging, subjective',
    scholarAgent: 'AI-powered precision',
  },
  {
    feature: 'Literature Review Generation',
    manual: 'Manual, laborious',
    scholarAgent: 'Automated, coherent synthesis',
  },
  {
    feature: 'Cost Efficiency',
    manual: 'High (time, labor)',
    scholarAgent: 'Optimized',
  },
  {
    feature: 'Real-time Updates',
    manual: false,
    scholarAgent: true,
  },
  {
    feature: 'Customization',
    manual: true,
    scholarAgent: true,
  },
];

const ComparisonTable: React.FC = () => {
  const renderComparisonValue = (value: string | boolean) => {
    if (typeof value === 'boolean') {
      return value ? (
        <CheckCircleOutlineIcon sx={{ color: 'var(--accent-success)', verticalAlign: 'middle', mr: 0.5 }} />
      ) : (
        <CancelOutlinedIcon sx={{ color: 'var(--accent-error)', verticalAlign: 'middle', mr: 0.5 }} />
      );
    }
    return value;
  };

  return (
    <TableContainer 
      component={Paper} 
      sx={{ 
        my: 4, 
        borderRadius: 'var(--radius-md)', 
        boxShadow: 'var(--shadow-lg)',
        backgroundColor: 'var(--bg-surface)',
        border: '1px solid var(--border-subtle)',
      }}
    >
      <Table aria-label="manual versus scholar agent comparison table">
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid var(--border-strong)', color: 'var(--text-primary)' }}>Feature</TableCell>
            <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid var(--border-strong)', color: 'var(--text-primary)' }}>Manual Research</TableCell>
            <TableCell sx={{ fontWeight: 'bold', borderBottom: '2px solid var(--border-strong)', color: 'var(--text-primary)' }}>Scholar Agent</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {comparisonData.map((row) => (
            <TableRow key={row.feature} sx={{ '&:hover': { backgroundColor: 'var(--bg-elevated)' } }}>
              <TableCell component="th" scope="row" sx={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <Typography variant="body1" sx={{ fontWeight: 'medium', color: 'var(--text-primary)' }}>{row.feature}</Typography>
              </TableCell>
              <TableCell sx={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-secondary)' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {renderComparisonValue(row.manual)}
                </Box>
              </TableCell>
              <TableCell sx={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--accent-success)' }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  {renderComparisonValue(row.scholarAgent)}
                </Box>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default ComparisonTable;
