import React, { useState, useMemo } from 'react';
import {
    Box,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Paper,
    TextField,
    InputAdornment,
    Chip,
    IconButton,
    Collapse,
    TableSortLabel,
    Button,
} from '@mui/material';
import { styled } from '@mui/system';
import SearchIcon from '@mui/icons-material/Search';
import FilterListIcon from '@mui/icons-material/FilterList';
import ArticleIcon from '@mui/icons-material/Article';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';
import TableChartIcon from '@mui/icons-material/TableChart';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import type { EvidenceMatrixRow } from '../../types';
import { formatAuthors } from '../../utils/exportEngine';

const MatrixContainer = styled(Paper)({
    backgroundColor: 'rgba(24, 24, 27, 0.7)',
    backdropFilter: 'blur(20px) saturate(180%)',
    WebkitBackdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    borderRadius: '16px',
    padding: '24px',
    boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
    color: '#F4F4F5',
});

const SearchInput = styled(TextField)({
    '& .MuiOutlinedInput-root': {
        backgroundColor: '#18181B',
        borderRadius: '10px',
        color: '#F4F4F5',
        fontSize: '0.875rem',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        '& fieldset': { border: 'none' },
        '&:hover': {
            backgroundColor: '#27272F',
        },
        '&.Mui-focused': {
            backgroundColor: '#18181B',
            border: '1px solid #FFB900',
        },
    },
    '& .MuiInputBase-input::placeholder': {
        color: '#71717A',
        opacity: 1,
    },
});

const StyledTableCell = styled(TableCell)({
    color: '#D4D4D8',
    borderColor: 'rgba(255, 255, 255, 0.06)',
    padding: '14px 16px',
    fontSize: '0.875rem',
});

const StyledHeaderCell = styled(TableCell)({
    color: '#A1A1AA',
    borderColor: 'rgba(255, 255, 255, 0.1)',
    fontWeight: 700,
    fontSize: '0.75rem',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    backgroundColor: 'rgba(18, 18, 20, 0.8)',
    '& .MuiTableSortLabel-root': {
        color: '#A1A1AA',
        '&:hover': { color: '#FFB900' },
        '&.Mui-active': {
            color: '#FFB900',
            '& .MuiTableSortLabel-icon': {
                color: '#FFB900 !important',
            },
        },
    },
});

const ExpandableRowBox = styled(Box)({
    backgroundColor: 'rgba(15, 15, 18, 0.95)',
    border: '1px solid rgba(255, 185, 0, 0.15)',
    borderRadius: '12px',
    padding: '20px',
    margin: '8px 16px 16px 16px',
});

type SortField = 'paper_id' | 'title' | 'year' | 'methodology' | 'benchmark_dataset' | 'primary_metric';
type SortOrder = 'asc' | 'desc';

interface EvidenceMatrixTableProps {
    rows: EvidenceMatrixRow[];
    title?: string;
    isLoading?: boolean;
}

export const EvidenceMatrixTable: React.FC<EvidenceMatrixTableProps> = ({
    rows = [],
    title = 'Comparative Evidence Matrix',
    isLoading = false,
}) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [fullTextOnly, setFullTextOnly] = useState(false);
    const [sortField, setSortField] = useState<SortField>('paper_id');
    const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
    const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

    const handleSort = (field: SortField) => {
        if (sortField === field) {
            setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortOrder('asc');
        }
    };

    const toggleRow = (paperId: string) => {
        setExpandedRows((prev) => {
            const next = new Set(prev);
            if (next.has(paperId)) {
                next.delete(paperId);
            } else {
                next.add(paperId);
            }
            return next;
        });
    };

    const fullTextCount = useMemo(() => {
        return rows.filter((r) => r.is_full_text || r.has_full_text).length;
    }, [rows]);

    const filteredAndSortedRows = useMemo(() => {
        return rows
            .filter((row) => {
                if (fullTextOnly && !(row.is_full_text || row.has_full_text)) {
                    return false;
                }
                if (!searchTerm.trim()) return true;
                const term = searchTerm.toLowerCase();
                return (
                    row.paper_id.toLowerCase().includes(term) ||
                    row.title.toLowerCase().includes(term) ||
                    (row.methodology || row.methodology_type || '').toLowerCase().includes(term) ||
                    (row.benchmark_dataset || row.dataset || '').toLowerCase().includes(term) ||
                    (row.primary_metric || '').toLowerCase().includes(term) ||
                    (row.primary_limitation || (row.limitations || []).join(' ')).toLowerCase().includes(term) ||
                    (row.authors || []).some((a) => a.toLowerCase().includes(term))
                );
            })
            .sort((a, b) => {
                let aVal = a[sortField] ?? '';
                let bVal = b[sortField] ?? '';
                if (typeof aVal === 'string') aVal = aVal.toLowerCase();
                if (typeof bVal === 'string') bVal = bVal.toLowerCase();

                if (aVal < bVal) return sortOrder === 'asc' ? -1 : 1;
                if (aVal > bVal) return sortOrder === 'asc' ? 1 : -1;
                return 0;
            });
    }, [rows, searchTerm, fullTextOnly, sortField, sortOrder]);

    return (
        <MatrixContainer elevation={0}>
            {/* Header / Title Bar */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3, flexWrap: 'wrap', gap: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    <TableChartIcon sx={{ color: '#FFB900', fontSize: '1.75rem' }} />
                    <Box>
                        <Typography variant="h5" sx={{ fontWeight: 800, color: '#F4F4F5', letterSpacing: '-0.02em' }}>
                            {title}
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#71717A' }}>
                            Structured comparative extraction across {rows.length} peer-reviewed scientific sources
                        </Typography>
                    </Box>
                </Box>

                {/* Filter and Search controls */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
                    <Chip
                        label={`All (${rows.length})`}
                        onClick={() => setFullTextOnly(false)}
                        sx={{
                            backgroundColor: !fullTextOnly ? 'rgba(255, 185, 0, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                            color: !fullTextOnly ? '#FFB900' : '#A1A1AA',
                            border: `1px solid ${!fullTextOnly ? 'rgba(255, 185, 0, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    />
                    <Chip
                        icon={<ArticleIcon sx={{ fontSize: '1rem !important', color: fullTextOnly ? '#00B894 !important' : 'inherit' }} />}
                        label={`Full-Text PDF (${fullTextCount})`}
                        onClick={() => setFullTextOnly(!fullTextOnly)}
                        sx={{
                            backgroundColor: fullTextOnly ? 'rgba(0, 184, 148, 0.15)' : 'rgba(255, 255, 255, 0.05)',
                            color: fullTextOnly ? '#00B894' : '#A1A1AA',
                            border: `1px solid ${fullTextOnly ? 'rgba(0, 184, 148, 0.4)' : 'rgba(255, 255, 255, 0.1)'}`,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    />
                    <SearchInput
                        size="small"
                        placeholder="Search matrix rows..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <SearchIcon sx={{ color: '#71717A', fontSize: '1.1rem' }} />
                                </InputAdornment>
                            ),
                        }}
                        sx={{ width: { xs: '100%', sm: '240px' } }}
                    />
                </Box>
            </Box>

            {/* Matrix Table */}
            {filteredAndSortedRows.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 6, color: '#71717A' }}>
                    <FilterListIcon sx={{ fontSize: '3rem', opacity: 0.3, mb: 1 }} />
                    <Typography variant="h6" sx={{ color: '#A1A1AA', fontWeight: 600 }}>
                        {isLoading ? 'Synthesizing evidence matrix...' : 'No matrix rows found matching criteria'}
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 0.5 }}>
                        {searchTerm ? 'Try adjusting your search keywords or clearing full-text filter.' : 'Matrix extraction will populate as papers are parsed.'}
                    </Typography>
                </Box>
            ) : (
                <TableContainer sx={{ border: '1px solid rgba(255, 255, 255, 0.08)', borderRadius: '12px', overflow: 'hidden' }}>
                    <Table sx={{ minWidth: 950 }} aria-label="evidence matrix">
                        <TableHead>
                            <TableRow>
                                <StyledHeaderCell width="40px" />
                                <StyledHeaderCell>
                                    <TableSortLabel
                                        active={sortField === 'paper_id'}
                                        direction={sortField === 'paper_id' ? sortOrder : 'asc'}
                                        onClick={() => handleSort('paper_id')}
                                    >
                                        Ref / Paper ID
                                    </TableSortLabel>
                                </StyledHeaderCell>
                                <StyledHeaderCell>
                                    <TableSortLabel
                                        active={sortField === 'title'}
                                        direction={sortField === 'title' ? sortOrder : 'asc'}
                                        onClick={() => handleSort('title')}
                                    >
                                        Title & Authors
                                    </TableSortLabel>
                                </StyledHeaderCell>
                                <StyledHeaderCell>
                                    <TableSortLabel
                                        active={sortField === 'methodology'}
                                        direction={sortField === 'methodology' ? sortOrder : 'asc'}
                                        onClick={() => handleSort('methodology')}
                                    >
                                        Methodology
                                    </TableSortLabel>
                                </StyledHeaderCell>
                                <StyledHeaderCell>
                                    <TableSortLabel
                                        active={sortField === 'benchmark_dataset'}
                                        direction={sortField === 'benchmark_dataset' ? sortOrder : 'asc'}
                                        onClick={() => handleSort('benchmark_dataset')}
                                    >
                                        Benchmark Dataset
                                    </TableSortLabel>
                                </StyledHeaderCell>
                                <StyledHeaderCell>
                                    <TableSortLabel
                                        active={sortField === 'primary_metric'}
                                        direction={sortField === 'primary_metric' ? sortOrder : 'asc'}
                                        onClick={() => handleSort('primary_metric')}
                                    >
                                        Primary Metric
                                    </TableSortLabel>
                                </StyledHeaderCell>
                                <StyledHeaderCell>Source</StyledHeaderCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {filteredAndSortedRows.map((row) => {
                                const isExpanded = expandedRows.has(row.paper_id);
                                const isFT = Boolean(row.is_full_text || row.has_full_text);
                                const authorText = formatAuthors(row.authors);
                                const yearText = row.year ? `(${row.year})` : '';

                                return (
                                    <React.Fragment key={row.paper_id}>
                                        <TableRow
                                            hover
                                            sx={{
                                                cursor: 'pointer',
                                                backgroundColor: isExpanded ? 'rgba(255, 185, 0, 0.03)' : 'transparent',
                                                '&:hover': { backgroundColor: 'rgba(255, 255, 255, 0.04)' },
                                                transition: 'background-color 0.2s ease',
                                            }}
                                            onClick={() => toggleRow(row.paper_id)}
                                        >
                                            <StyledTableCell>
                                                <IconButton size="small" sx={{ color: '#A1A1AA' }}>
                                                    {isExpanded ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
                                                </IconButton>
                                            </StyledTableCell>
                                            <StyledTableCell sx={{ fontWeight: 700, color: '#FFB900', fontFamily: 'var(--font-mono)' }}>
                                                [{row.paper_id}]
                                            </StyledTableCell>
                                            <StyledTableCell sx={{ maxWidth: '280px' }}>
                                                <Typography variant="body2" sx={{ fontWeight: 700, color: '#F4F4F5', lineHeight: 1.3 }}>
                                                    {row.title}
                                                </Typography>
                                                <Typography variant="caption" sx={{ color: '#A1A1AA', display: 'block', mt: 0.5 }}>
                                                    {authorText} {yearText}
                                                </Typography>
                                            </StyledTableCell>
                                            <StyledTableCell sx={{ maxWidth: '220px' }}>
                                                <Typography variant="body2" sx={{ color: '#00F5C8', fontWeight: 600, fontSize: '0.82rem' }}>
                                                    {row.methodology || row.methodology_type || '-'}
                                                </Typography>
                                            </StyledTableCell>
                                            <StyledTableCell sx={{ maxWidth: '180px' }}>
                                                <Typography variant="body2" sx={{ color: '#D4D4D8', fontSize: '0.82rem' }}>
                                                    {row.benchmark_dataset || row.dataset || '-'}
                                                </Typography>
                                            </StyledTableCell>
                                            <StyledTableCell sx={{ maxWidth: '180px' }}>
                                                <Chip
                                                    label={row.primary_metric || 'Evaluated'}
                                                    size="small"
                                                    sx={{
                                                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                                                        color: '#38BDF8',
                                                        fontWeight: 700,
                                                        fontSize: '0.75rem',
                                                        border: '1px solid rgba(56, 189, 248, 0.3)',
                                                    }}
                                                />
                                            </StyledTableCell>
                                            <StyledTableCell>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <Chip
                                                        label={isFT ? 'Full Text' : 'Abstract'}
                                                        size="small"
                                                        sx={{
                                                            backgroundColor: isFT ? 'rgba(0, 184, 148, 0.15)' : 'rgba(161, 161, 170, 0.1)',
                                                            color: isFT ? '#00B894' : '#A1A1AA',
                                                            fontWeight: 700,
                                                            fontSize: '0.7rem',
                                                            border: `1px solid ${isFT ? 'rgba(0, 184, 148, 0.3)' : 'rgba(161, 161, 170, 0.2)'}`,
                                                        }}
                                                    />
                                                    {(row.url || row.doi) && (
                                                        <IconButton
                                                            size="small"
                                                            component="a"
                                                            href={row.url || (row.doi ? `https://doi.org/${row.doi}` : '#')}
                                                            target="_blank"
                                                            rel="noopener noreferrer"
                                                            onClick={(e) => e.stopPropagation()}
                                                            sx={{
                                                                color: '#FFB900',
                                                                p: 0.5,
                                                                '&:hover': { color: '#FFE082', backgroundColor: 'rgba(255, 185, 0, 0.15)' },
                                                            }}
                                                            title="Open paper source"
                                                        >
                                                            <OpenInNewIcon sx={{ fontSize: '0.95rem' }} />
                                                        </IconButton>
                                                    )}
                                                </Box>
                                            </StyledTableCell>
                                        </TableRow>

                                        {/* Expandable Limitations & Details Row */}
                                        <TableRow>
                                            <TableCell style={{ paddingBottom: 0, paddingTop: 0, border: 0 }} colSpan={7}>
                                                <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                                                    <ExpandableRowBox>
                                                        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3 }}>
                                                            {/* Primary Limitations */}
                                                            <Box>
                                                                <Typography
                                                                    variant="subtitle2"
                                                                    sx={{ color: '#EF4444', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', mb: 1 }}
                                                                >
                                                                    Identified Limitations & Bottlenecks
                                                                </Typography>
                                                                <Typography variant="body2" sx={{ color: '#E4E4E7', lineHeight: 1.6, mb: 1.5 }}>
                                                                    {row.primary_limitation || (row.limitations && row.limitations.length > 0 ? row.limitations.join('; ') : 'No specific limitations noted.')}
                                                                </Typography>

                                                                {row.limitations && row.limitations.length > 1 && (
                                                                    <Box sx={{ pl: 2 }}>
                                                                        {row.limitations.map((lim, idx) => (
                                                                            <Typography key={idx} variant="caption" sx={{ color: '#A1A1AA', display: 'block', mb: 0.5 }}>
                                                                                • {lim}
                                                                            </Typography>
                                                                        ))}
                                                                    </Box>
                                                                )}
                                                            </Box>

                                                            {/* Methodological Context & Findings */}
                                                            <Box>
                                                                <Typography
                                                                    variant="subtitle2"
                                                                    sx={{ color: '#00F5C8', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', mb: 1 }}
                                                                >
                                                                    Benchmark & Quantitative Results
                                                                </Typography>
                                                                <Box sx={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', p: 1.5, borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)', mb: 1.5 }}>
                                                                    <Typography variant="body2" sx={{ color: '#D4D4D8', mb: 0.5 }}>
                                                                        <strong>Benchmark:</strong> {row.benchmark_dataset || row.dataset || 'Standard Corpus'}
                                                                    </Typography>
                                                                    <Typography variant="body2" sx={{ color: '#D4D4D8' }}>
                                                                        <strong>Primary Metric:</strong> {row.primary_metric || 'Evaluation Metric Achieved'}
                                                                    </Typography>
                                                                </Box>

                                                                {row.key_findings && row.key_findings.length > 0 && (
                                                                    <Box>
                                                                        <Typography variant="caption" sx={{ color: '#71717A', fontWeight: 700, textTransform: 'uppercase', display: 'block', mb: 0.5 }}>
                                                                            Key Findings:
                                                                        </Typography>
                                                                        {row.key_findings.map((finding, idx) => (
                                                                            <Typography key={idx} variant="caption" sx={{ color: '#A1A1AA', display: 'block', mb: 0.5 }}>
                                                                                ✓ {finding}
                                                                            </Typography>
                                                                        ))}
                                                                    </Box>
                                                                )}
                                                            </Box>
                                                        </Box>

                                                        {(row.url || row.doi) && (
                                                            <Box sx={{ mt: 2, pt: 1.5, borderTop: '1px solid rgba(255, 255, 255, 0.08)', display: 'flex', flexWrap: 'wrap', gap: 1.5, alignItems: 'center' }}>
                                                                <Button
                                                                    component="a"
                                                                    href={row.url || (row.doi ? `https://doi.org/${row.doi}` : '#')}
                                                                    target="_blank"
                                                                    rel="noopener noreferrer"
                                                                    size="small"
                                                                    startIcon={<OpenInNewIcon sx={{ fontSize: '0.85rem !important' }} />}
                                                                    sx={{
                                                                        backgroundColor: 'rgba(255, 185, 0, 0.1)',
                                                                        color: '#FFB900',
                                                                        border: '1px solid rgba(255, 185, 0, 0.3)',
                                                                        textTransform: 'none',
                                                                        fontWeight: 700,
                                                                        fontSize: '0.75rem',
                                                                        borderRadius: '8px',
                                                                        py: 0.4,
                                                                        px: 1.2,
                                                                        '&:hover': {
                                                                            backgroundColor: 'rgba(255, 185, 0, 0.2)',
                                                                            borderColor: 'rgba(255, 185, 0, 0.5)',
                                                                        },
                                                                    }}
                                                                >
                                                                    View Paper Source
                                                                </Button>
                                                                {row.doi && (
                                                                    <Button
                                                                        component="a"
                                                                        href={`https://doi.org/${row.doi.replace(/^https?:\/\/doi\.org\//, '')}`}
                                                                        target="_blank"
                                                                        rel="noopener noreferrer"
                                                                        size="small"
                                                                        startIcon={<OpenInNewIcon sx={{ fontSize: '0.85rem !important' }} />}
                                                                        sx={{
                                                                            backgroundColor: 'rgba(56, 189, 248, 0.1)',
                                                                            color: '#38BDF8',
                                                                            border: '1px solid rgba(56, 189, 248, 0.3)',
                                                                            textTransform: 'none',
                                                                            fontWeight: 700,
                                                                            fontSize: '0.75rem',
                                                                            borderRadius: '8px',
                                                                            py: 0.4,
                                                                            px: 1.2,
                                                                            '&:hover': {
                                                                                backgroundColor: 'rgba(56, 189, 248, 0.2)',
                                                                                borderColor: 'rgba(56, 189, 248, 0.5)',
                                                                            },
                                                                        }}
                                                                    >
                                                                        DOI Link
                                                                    </Button>
                                                                )}
                                                            </Box>
                                                        )}
                                                    </ExpandableRowBox>
                                                </Collapse>
                                            </TableCell>
                                        </TableRow>
                                    </React.Fragment>
                                );
                            })}
                        </TableBody>
                    </Table>
                </TableContainer>
            )}
        </MatrixContainer>
    );
};

export default EvidenceMatrixTable;