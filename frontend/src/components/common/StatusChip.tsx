import React from 'react';
import type { ResearchProject } from '../../types';

interface StatusChipProps {
  status: ResearchProject['status'];
}

const statusStyles: { [key in ResearchProject['status']]: string } = {
  creating: 'bg-gray-200 text-gray-800 animate-pulse',
  planning: 'bg-blue-100 text-blue-800',
  created: 'bg-gray-100 text-gray-800',
  searching: 'bg-cyan-100 text-cyan-800 animate-pulse',
  analyzing: 'bg-indigo-100 text-indigo-800 animate-pulse',
  synthesizing: 'bg-purple-100 text-purple-800 animate-pulse',
  completed: 'bg-green-100 text-green-800',
  error: 'bg-red-100 text-red-800',
  error_no_papers_found: 'bg-yellow-100 text-yellow-800',
};

const statusText: { [key in ResearchProject['status']]: string } = {
    creating: 'Creating...',
    planning: 'Planning',
    created: 'Ready to Start',
    searching: 'Searching...',
    analyzing: 'Analyzing...',
    synthesizing: 'Synthesizing...',
    completed: 'Completed',
    error: 'Error',
    error_no_papers_found: 'No Papers Found',
}

const StatusChip: React.FC<StatusChipProps> = ({ status }) => {
  const style = statusStyles[status] || 'bg-gray-100 text-blue-500';
  const text = statusText[status] || 'Unknown';

  return (
  <span
    className={`px-3 py-1 text-xs font-medium inline-block ${
      text === 'Ready to Start' ? 'rounded-xl border' : 'rounded-full'
    } ${style}`}
  >
    {text}
  </span>
);
};

export default StatusChip;