import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { neonData } from "../api/neonClient";
import {
  Typography,
  Box,
  CircularProgress,
  Alert,
  Button,
  Menu,
  MenuItem,
  Tabs,
  Tab,
  Chip,
  LinearProgress,
  IconButton,
  Divider,
} from "@mui/material";
import { styled } from "@mui/system";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import jsPDF from "jspdf";
import { useState } from "react";
import { unified } from "unified";
import remarkParse from "remark-parse";
import {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  BorderStyle,
} from "docx";
import type { Root, Content } from "mdast";
import type { PaperReference } from "../types";
import { saveAs } from "file-saver";
import DownloadIcon from "@mui/icons-material/Download";
import ShareIcon from "@mui/icons-material/Share";
import MoreVertIcon from "@mui/icons-material/MoreVert";
import SchoolIcon from "@mui/icons-material/School";
import AnalyticsIcon from "@mui/icons-material/Analytics";
import TimelineIcon from "@mui/icons-material/Timeline";
import LibraryBooksIcon from "@mui/icons-material/LibraryBooks";
import SettingsIcon from "@mui/icons-material/Settings";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";

type HeadingValue = (typeof HeadingLevel)[keyof typeof HeadingLevel];

// ==========================================
// STYLED COMPONENTS - NEO-MODERN RESEARCH NOIR
// ==========================================

const PageWrapper = styled(Box)({
  minHeight: "100vh",
  backgroundColor: "#09090B",
  backgroundImage:
    "radial-gradient(circle at 50% -20%, rgba(255, 185, 0, 0.04) 0%, transparent 50%)",
  color: "#F4F4F5",
  paddingTop: "100px",
  paddingBottom: "80px",
});

const GlassCard = styled(Box)({
  background: "rgba(24, 24, 27, 0.7)",
  backdropFilter: "blur(20px) saturate(180%)",
  WebkitBackdropFilter: "blur(20px) saturate(180%)",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  borderRadius: "16px",
  boxShadow: "0 8px 32px 0 rgba(0, 0, 0, 0.3)",
  transition: "all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1)",
});

const HeaderCard = styled(GlassCard)({
  padding: "2rem",
  marginBottom: "2rem",
  background:
    "linear-gradient(135deg, rgba(24, 24, 27, 0.85) 0%, rgba(39, 39, 47, 0.75) 100%)",
  borderRadius: "20px",
  position: "relative",
  overflow: "hidden",
  "&::before": {
    content: '""',
    position: "absolute",
    top: 0,
    right: 0,
    width: "300px",
    height: "300px",
    background:
      "radial-gradient(circle, rgba(255, 185, 0, 0.08) 0%, transparent 70%)",
    pointerEvents: "none",
  },
});

const StyledTabs = styled(Tabs)({
  marginBottom: "2rem",
  backgroundColor: "rgba(24, 24, 27, 0.5)",
  borderRadius: "12px",
  padding: "8px",
  border: "1px solid rgba(255, 255, 255, 0.05)",
  "& .MuiTabs-indicator": {
    backgroundColor: "#FFB900",
    height: "3px",
    borderRadius: "3px 3px 0 0",
  },
});

const StyledTab = styled(Tab)({
  color: "#A1A1AA",
  fontWeight: 600,
  fontSize: "0.9rem",
  textTransform: "none",
  minHeight: "48px",
  transition: "all 0.2s ease",
  borderRadius: "8px",
  "&:hover": {
    color: "#F4F4F5",
    backgroundColor: "rgba(255, 255, 255, 0.05)",
  },
  "&.Mui-selected": {
    color: "#FFB900",
  },
  "& .MuiTab-iconWrapper": {
    marginBottom: "4px",
  },
});

const PaperCard = styled(GlassCard)({
  padding: "1.5rem",
  transition:
    "all 0.3s cubic-bezier(0.4, 0.0, 0.2, 1), transform 0.2s cubic-bezier(0.68, -0.55, 0.265, 1.55)",
  cursor: "pointer",
  "&:hover": {
    transform: "translateY(-4px)",
    borderColor: "rgba(0, 245, 200, 0.3)",
    boxShadow: "0 12px 40px rgba(0, 245, 200, 0.1)",
  },
});

const StatusBadge = styled(Chip)<{ status?: string }>(({ status }) => {
  const statusColors: Record<string, { bg: string; text: string; glow: string }> = {
    completed: {
      bg: "rgba(0, 184, 148, 0.15)",
      text: "#00B894",
      glow: "0 0 20px rgba(0, 184, 148, 0.3)",
    },
    processing: {
      bg: "rgba(255, 185, 0, 0.15)",
      text: "#FFB900",
      glow: "0 0 20px rgba(255, 185, 0, 0.3)",
    },
    error: {
      bg: "rgba(244, 67, 54, 0.15)",
      text: "#F44336",
      glow: "0 0 20px rgba(244, 67, 54, 0.3)",
    },
    pending: {
      bg: "rgba(161, 161, 170, 0.15)",
      text: "#A1A1AA",
      glow: "none",
    },
  };

  const colors = statusColors[status || "pending"] || statusColors.pending;

  return {
    backgroundColor: colors.bg,
    color: colors.text,
    fontWeight: 700,
    fontSize: "0.75rem",
    height: "28px",
    borderRadius: "14px",
    border: `1px solid ${colors.text}40`,
    boxShadow: colors.glow,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
  };
});

const ActionButton = styled(Button)({
  backgroundColor: "rgba(255, 255, 255, 0.05)",
  color: "#F4F4F5",
  border: "1px solid rgba(255, 255, 255, 0.1)",
  borderRadius: "10px",
  padding: "10px 20px",
  fontWeight: 600,
  textTransform: "none",
  transition: "all 0.2s ease",
  "&:hover": {
    backgroundColor: "rgba(255, 255, 255, 0.1)",
    borderColor: "rgba(255, 185, 0, 0.5)",
    transform: "translateY(-2px)",
  },
});

const PrimaryButton = styled(Button)({
  backgroundColor: "#FFB900",
  color: "#09090B",
  borderRadius: "10px",
  padding: "10px 24px",
  fontWeight: 700,
  textTransform: "none",
  boxShadow: "0 4px 14px rgba(255, 185, 0, 0.3)",
  transition: "all 0.2s ease",
  "&:hover": {
    backgroundColor: "#E6A600",
    transform: "translateY(-2px)",
    boxShadow: "0 6px 20px rgba(255, 185, 0, 0.4)",
  },
});

const EmptyState = styled(Box)({
  textAlign: "center",
  padding: "4rem 2rem",
  color: "#71717A",
  "& svg": {
    fontSize: "4rem",
    opacity: 0.3,
    marginBottom: "1rem",
  },
});

const RelevanceScore = styled(Box)<{ score?: number }>(({ score = 0 }) => {
  const getColor = () => {
    if (score >= 80) return "#00B894"; // Discovery Green
    if (score >= 60) return "#00F5C8"; // Aurora Teal
    if (score >= 40) return "#FFB900"; // Insight Amber
    return "#71717A"; // Neutral
  };

  return {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: `${getColor()}20`,
    color: getColor(),
    fontWeight: 700,
    fontSize: "0.85rem",
    padding: "6px 12px",
    borderRadius: "8px",
    border: `1.5px solid ${getColor()}40`,
  };
});

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

const extractText = (node: Content): string => {
  if (node.type === "text") {
    return node.value;
  }
  if ("children" in node && Array.isArray(node.children)) {
    return node.children.map((child) => extractText(child as Content)).join("");
  }
  return "";
};

const getHeadingLevel = (depth: number): HeadingValue => {
  const levels: Record<number, HeadingValue> = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3,
    4: HeadingLevel.HEADING_4,
    5: HeadingLevel.HEADING_5,
    6: HeadingLevel.HEADING_6,
  };
  return levels[depth] || HeadingLevel.HEADING_6;
};

const markdownToDocxElements = (markdown: string): Paragraph[] => {
  const tree = unified().use(remarkParse).parse(markdown) as Root;
  const elements: Paragraph[] = [];

  const processNode = (node: Content) => {
    switch (node.type) {
      case "heading": {
        const text = extractText(node);
        const level = getHeadingLevel(node.depth);
        elements.push(new Paragraph({ text, heading: level }));
        break;
      }
      case "paragraph": {
        const runs: TextRun[] = [];
        node.children.forEach((child) => {
          const text = extractText(child as Content);
          if (child.type === "strong") {
            runs.push(new TextRun({ text, bold: true }));
          } else if (child.type === "emphasis") {
            runs.push(new TextRun({ text, italics: true }));
          } else {
            runs.push(new TextRun(text));
          }
        });
        elements.push(new Paragraph({ children: runs }));
        break;
      }
      case "list": {
        node.children.forEach((listItem) => {
          const text = extractText(listItem);
          elements.push(
            new Paragraph({
              text: text,
              bullet: { level: 0 },
            })
          );
        });
        break;
      }
      case "thematicBreak": {
        elements.push(
          new Paragraph({
            border: {
              bottom: {
                color: "auto",
                space: 1,
                style: BorderStyle.SINGLE,
                size: 6,
              },
            },
          })
        );
        break;
      }
      default:
        break;
    }
  };

  tree.children.forEach(processNode);
  return elements;
};

// ==========================================
// MAIN COMPONENT
// ==========================================

const ProjectDetailsPage = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const [activeTab, setActiveTab] = useState(0);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [exportMenuAnchor, setExportMenuAnchor] = useState<null | HTMLElement>(
    null
  );
  const open = Boolean(anchorEl);
  const exportMenuOpen = Boolean(exportMenuAnchor);

  const {
    data: project,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["project", projectId],
    queryFn: () => neonData.getProjectById(projectId!),
    enabled: !!projectId,
  });

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const handleMenuClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleExportClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setExportMenuAnchor(event.currentTarget);
  };

  const handleExportClose = () => {
    setExportMenuAnchor(null);
  };

  const exportToPdf = () => {
    const input = document.getElementById("synthesis-output");
    if (input) {
      const pdf = new jsPDF("p", "mm", "a4");
      pdf.html(input, {
        callback: function (doc) {
          doc.save("literature-review.pdf");
          handleExportClose();
        },
        margin: [15, 15, 15, 15],
        autoPaging: "text",
        width: 180,
        windowWidth: 1000,
      });
    }
  };

  const exportToDocx = () => {
    const synthesizerPlan = project?.agent_plans.find(
      (p) => p.agent_type === "synthesizer"
    );
    const synthesisOutput =
      (typeof synthesizerPlan?.plan_steps[0]?.output.response === "string"
        ? synthesizerPlan?.plan_steps[0]?.output.response
        : "") || "No synthesis available.";

    const docElements = markdownToDocxElements(synthesisOutput);

    const doc = new Document({
      sections: [
        {
          children: docElements,
        },
      ],
    });

    Packer.toBlob(doc).then((blob) => {
      saveAs(blob, "literature-review.docx");
    });

    handleExportClose();
  };

  if (isLoading) {
    return (
      <PageWrapper>
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: "60vh",
            flexDirection: "column",
            gap: 2,
          }}
        >
          <CircularProgress sx={{ color: "#FFB900" }} size={50} />
          <Typography sx={{ color: "#71717A", fontFamily: "var(--font-content)" }}>
            Loading project details...
          </Typography>
        </Box>
      </PageWrapper>
    );
  }

  if (error) {
    return (
      <PageWrapper>
        <Box sx={{ px: { xs: 2, sm: 3, md: 6 }, py: 4, maxWidth: "800px", mx: "auto" }}>
          <Alert
            severity="error"
            sx={{
              backgroundColor: "rgba(244, 67, 54, 0.15)",
              color: "#F44336",
              border: "1px solid rgba(244, 67, 54, 0.3)",
              borderRadius: "12px",
            }}
          >
            Failed to load project details: {error.message}
          </Alert>
        </Box>
      </PageWrapper>
    );
  }

  if (!project) {
    return (
      <PageWrapper>
        <Box sx={{ px: { xs: 2, sm: 3, md: 6 }, py: 4, maxWidth: "800px", mx: "auto" }}>
          <Alert
            severity="warning"
            sx={{
              backgroundColor: "rgba(255, 185, 0, 0.15)",
              color: "#FFB900",
              border: "1px solid rgba(255, 185, 0, 0.3)",
              borderRadius: "12px",
            }}
          >
            Project not found.
          </Alert>
        </Box>
      </PageWrapper>
    );
  }

  const synthesizerPlan = project.agent_plans.find(
    (p) => p.agent_type === "synthesizer"
  );
  const synthesisOutput =
    (typeof synthesizerPlan?.plan_steps[0]?.output.response === "string"
      ? synthesizerPlan?.plan_steps[0]?.output.response
      : "") || "No synthesis available.";

  const projectStatus = project.status || "completed";

  // ==========================================
  // TAB PANELS
  // ==========================================

  const renderTabContent = () => {
    switch (activeTab) {
      case 0: // Literature Review
        return (
          <GlassCard sx={{ p: { xs: 2, sm: 3, md: 4 } }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 800,
                  color: "#F4F4F5",
                  letterSpacing: "-0.02em",
                }}
              >
                Synthesized Literature Review
              </Typography>
              <PrimaryButton
                startIcon={<DownloadIcon />}
                onClick={handleExportClick}
                size="small"
              >
                Export
              </PrimaryButton>
            </Box>
            <Divider sx={{ mb: 3, borderColor: "rgba(255, 255, 255, 0.1)" }} />
            <Box
              id="synthesis-output"
              sx={{
                "& h1": {
                  color: "#F4F4F5",
                  fontSize: { xs: "1.5rem", sm: "1.75rem", md: "2rem" },
                  fontWeight: 700,
                  marginTop: "1.5rem",
                  marginBottom: "1rem",
                },
                "& h2": {
                  color: "#F4F4F5",
                  fontSize: { xs: "1.25rem", sm: "1.5rem" },
                  fontWeight: 700,
                  marginTop: "1.25rem",
                  marginBottom: "0.75rem",
                },
                "& h3": {
                  color: "#E4E4E7",
                  fontSize: { xs: "1.1rem", sm: "1.25rem" },
                  fontWeight: 600,
                  marginTop: "1rem",
                  marginBottom: "0.5rem",
                },
                "& p": {
                  color: "#D4D4D8",
                  fontSize: { xs: "0.95rem", sm: "1rem" },
                  lineHeight: 1.8,
                  marginBottom: "1rem",
                  fontFamily: "var(--font-content)",
                },
                "& li": {
                  color: "#D4D4D8",
                  fontSize: { xs: "0.95rem", sm: "1rem" },
                  lineHeight: 1.7,
                  marginBottom: "0.5rem",
                  fontFamily: "var(--font-content)",
                },
                "& a": {
                  color: "#00F5C8",
                  textDecoration: "none",
                  borderBottom: "1px solid rgba(0, 245, 200, 0.3)",
                  transition: "all 0.2s ease",
                  "&:hover": {
                    color: "#FFB900",
                    borderBottomColor: "rgba(255, 185, 0, 0.5)",
                  },
                },
                "& code": {
                  backgroundColor: "rgba(255, 185, 0, 0.1)",
                  color: "#FFB900",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  fontSize: "0.9em",
                  fontFamily: "var(--font-mono)",
                },
                "& pre": {
                  backgroundColor: "rgba(24, 24, 27, 0.5)",
                  padding: "1rem",
                  borderRadius: "8px",
                  overflow: "auto",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                },
              }}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {synthesisOutput}
              </ReactMarkdown>
            </Box>
          </GlassCard>
        );

      case 1: // Discovered Papers
        return (
          <Box>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 800,
                  color: "#F4F4F5",
                  letterSpacing: "-0.02em",
                }}
              >
                Discovered Papers ({project.paper_references.length})
              </Typography>
              <Chip
                icon={<LibraryBooksIcon sx={{ fontSize: "1rem" }} />}
                label={`${project.paper_references.length} Papers`}
                sx={{
                  backgroundColor: "rgba(0, 245, 200, 0.15)",
                  color: "#00F5C8",
                  fontWeight: 600,
                  border: "1px solid rgba(0, 245, 200, 0.3)",
                }}
              />
            </Box>

            {project.paper_references.length === 0 ? (
              <EmptyState>
                <SchoolIcon />
                <Typography variant="h6" sx={{ color: "#A1A1AA", mb: 1 }}>
                  No papers discovered yet
                </Typography>
                <Typography variant="body2">
                  Papers will appear here as the agents discover them.
                </Typography>
              </EmptyState>
            ) : (
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)" },
                  gap: 3,
                }}
              >
                {project.paper_references
                  .sort((a, b) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0))
                  .map((paper: PaperReference) => (
                    <PaperCard key={paper.id}>
                      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", mb: 2 }}>
                        <Typography
                          variant="h6"
                          sx={{
                            fontWeight: 700,
                            color: "#F4F4F5",
                            fontSize: { xs: "1rem", sm: "1.125rem" },
                            flex: 1,
                            pr: 2,
                            lineHeight: 1.4,
                          }}
                        >
                          {paper.title}
                        </Typography>
                        <RelevanceScore score={paper.relevance_score}>
                          {paper.relevance_score.toFixed(0)}
                        </RelevanceScore>
                      </Box>

                      <Typography
                        variant="body2"
                        sx={{
                          color: "#A1A1AA",
                          fontSize: "0.85rem",
                          mb: 2,
                          fontStyle: "italic",
                        }}
                      >
                        {paper.authors.join(", ")}
                      </Typography>

                      <Box sx={{ display: "flex", gap: 1, mt: 2 }}>
                        <Button
                          component="a"
                          href={paper.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          size="small"
                          sx={{
                            backgroundColor: "rgba(0, 245, 200, 0.1)",
                            color: "#00F5C8",
                            border: "1px solid rgba(0, 245, 200, 0.3)",
                            textTransform: "none",
                            fontWeight: 600,
                            fontSize: "0.8rem",
                            borderRadius: "8px",
                            "&:hover": {
                              backgroundColor: "rgba(0, 245, 200, 0.2)",
                              borderColor: "rgba(0, 245, 200, 0.5)",
                            },
                          }}
                        >
                          Read Paper
                        </Button>
                      </Box>
                    </PaperCard>
                  ))}
              </Box>
            )}
          </Box>
        );

      case 2: // Analysis Results
        return (
          <EmptyState>
            <AnalyticsIcon />
            <Typography variant="h6" sx={{ color: "#A1A1AA", mb: 1 }}>
              Analysis Results Coming Soon
            </Typography>
            <Typography variant="body2">
              Detailed analysis results will be available in a future update.
            </Typography>
          </EmptyState>
        );

      case 3: // Research Gaps
        return (
          <EmptyState>
            <TrendingUpIcon />
            <Typography variant="h6" sx={{ color: "#A1A1AA", mb: 1 }}>
              Research Gaps Coming Soon
            </Typography>
            <Typography variant="body2">
              Identified research gaps will appear here after synthesis.
            </Typography>
          </EmptyState>
        );

      case 4: // Live Progress
        return (
          <EmptyState>
            <TimelineIcon />
            <Typography variant="h6" sx={{ color: "#A1A1AA", mb: 1 }}>
              Live Progress Coming Soon
            </Typography>
            <Typography variant="body2">
              Real-time agent activity will be displayed here.
            </Typography>
          </EmptyState>
        );

      case 5: // Settings
        return (
          <EmptyState>
            <SettingsIcon />
            <Typography variant="h6" sx={{ color: "#A1A1AA", mb: 1 }}>
              Project Settings Coming Soon
            </Typography>
            <Typography variant="body2">
              Configure project settings and preferences here.
            </Typography>
          </EmptyState>
        );

      default:
        return null;
    }
  };

  return (
    <PageWrapper>
      <Box sx={{ px: { xs: 2, sm: 3, md: 6 }, maxWidth: "1400px", mx: "auto" }}>
        {/* Header Actions */}
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 3,
          }}
        >
          <Button
            component={Link}
            to="/"
            startIcon={<ArrowBackIcon />}
            sx={{
              color: "#A1A1AA",
              fontWeight: 600,
              textTransform: "none",
              "&:hover": {
                color: "#F4F4F5",
                backgroundColor: "rgba(255, 255, 255, 0.05)",
              },
            }}
          >
            
          </Button>

          <Box sx={{ display: "flex", gap: 1 }}>
            <ActionButton startIcon={<ShareIcon />} size="small">
              Share
            </ActionButton>
            <IconButton
              onClick={handleMenuClick}
              sx={{
                color: "#A1A1AA",
                "&:hover": { backgroundColor: "rgba(255, 255, 255, 0.05)" },
              }}
            >
              <MoreVertIcon />
            </IconButton>
          </Box>
        </Box>

        {/* Project Header */}
        <HeaderCard>
          <Box sx={{ position: "relative", zIndex: 1 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", mb: 2 }}>
              <Box sx={{ flex: 1 }}>
                <Typography
                  variant="h3"
                  sx={{
                    fontWeight: 800,
                    color: "#F4F4F5",
                    letterSpacing: "-0.03em",
                    fontSize: { xs: "1.75rem", sm: "2.25rem", md: "2.5rem" },
                    mb: 1,
                  }}
                >
                  {project.title}
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    color: "#A1A1AA",
                    fontFamily: "var(--font-content)",
                    fontStyle: "italic",
                    fontSize: { xs: "1rem", sm: "1.125rem" },
                    fontWeight: 400,
                  }}
                >
                  "{project.research_question}"
                </Typography>
              </Box>
              <StatusBadge label={projectStatus} status={projectStatus} />
            </Box>

            {/* Progress Bar */}
            <Box sx={{ mt: 3 }}>
              <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
                <Typography variant="body2" sx={{ color: "#71717A", fontWeight: 600 }}>
                  Overall Progress
                </Typography>
                <Typography variant="body2" sx={{ color: "#FFB900", fontWeight: 700 }}>
                  100%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={100}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: "rgba(255, 255, 255, 0.1)",
                  "& .MuiLinearProgress-bar": {
                    borderRadius: 4,
                    background: "linear-gradient(90deg, #00B894 0%, #00F5C8 50%, #FFB900 100%)",
                  },
                }}
              />
            </Box>
          </Box>
        </HeaderCard>

        {/* Tabs Navigation */}
        <StyledTabs
          value={activeTab}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          allowScrollButtonsMobile
        >
          <StyledTab icon={<LibraryBooksIcon />} iconPosition="start" label="Review" />
          <StyledTab icon={<SchoolIcon />} iconPosition="start" label="Papers" />
          <StyledTab icon={<AnalyticsIcon />} iconPosition="start" label="Analysis" />
          <StyledTab icon={<TrendingUpIcon />} iconPosition="start" label="Gaps" />
          <StyledTab icon={<TimelineIcon />} iconPosition="start" label="Progress" />
          <StyledTab icon={<SettingsIcon />} iconPosition="start" label="Settings" />
        </StyledTabs>

        {/* Tab Content */}
        <Box>{renderTabContent()}</Box>

        {/* Export Menu */}
        <Menu
          anchorEl={exportMenuAnchor}
          open={exportMenuOpen}
          onClose={handleExportClose}
          PaperProps={{
            sx: {
              backgroundColor: "#18181B",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: "12px",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
            },
          }}
        >
          <MenuItem
            onClick={exportToPdf}
            sx={{
              color: "#F4F4F5",
              "&:hover": { backgroundColor: "rgba(255, 255, 255, 0.1)" },
            }}
          >
            Export as PDF
          </MenuItem>
          <MenuItem
            onClick={exportToDocx}
            sx={{
              color: "#F4F4F5",
              "&:hover": { backgroundColor: "rgba(255, 255, 255, 0.1)" },
            }}
          >
            Export as DOCX
          </MenuItem>
        </Menu>

        {/* More Menu */}
        <Menu
          anchorEl={anchorEl}
          open={open}
          onClose={handleMenuClose}
          PaperProps={{
            sx: {
              backgroundColor: "#18181B",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: "12px",
              boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
            },
          }}
        >
          <MenuItem
            onClick={handleMenuClose}
            sx={{
              color: "#F4F4F5",
              "&:hover": { backgroundColor: "rgba(255, 255, 255, 0.1)" },
            }}
          >
            Duplicate Project
          </MenuItem>
          <MenuItem
            onClick={handleMenuClose}
            sx={{
              color: "#F44336",
              "&:hover": { backgroundColor: "rgba(244, 67, 54, 0.1)" },
            }}
          >
            Delete Project
          </MenuItem>
        </Menu>
      </Box>
    </PageWrapper>
  );
};

export default ProjectDetailsPage;
