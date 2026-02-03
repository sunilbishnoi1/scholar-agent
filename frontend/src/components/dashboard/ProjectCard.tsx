import React, { useState } from "react";
import {
  Card,
  CardContent,
  Typography,
  Button,
  Box,
  LinearProgress,
  CircularProgress,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Tooltip,
  Chip,
  Divider,
} from "@mui/material";
import { Link, useNavigate } from "react-router-dom";
import type { ResearchProject } from "../../types";
import StatusChip from "../common/StatusChip";
import { startLiteratureReview } from "../../api/client";
import { useProjectStore } from "../../store/projectStore";
import { toast } from "react-toastify";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import ArticleOutlinedIcon from "@mui/icons-material/ArticleOutlined";
import CalendarTodayOutlinedIcon from "@mui/icons-material/CalendarTodayOutlined";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import RefreshIcon from "@mui/icons-material/Refresh";
import { useProjectStream } from "../../hooks/useProjectStream";

interface ProjectCardProps {
  project: ResearchProject;
}

const ProgressTracker: React.FC<{
  project: ResearchProject;
  realtimePapersAnalyzed?: number;
  realtimeTotalPapers?: number;
}> = ({ project, realtimePapersAnalyzed, realtimeTotalPapers }) => {
  // Prefer real-time WebSocket counts over static project data
  const papersAnalyzed =
    realtimePapersAnalyzed !== undefined
      ? realtimePapersAnalyzed
      : project.agent_plans.filter((p) => p.agent_type === "analyzer").length;
  const totalPapersToAnalyze =
    realtimeTotalPapers !== undefined && realtimeTotalPapers > 0
      ? realtimeTotalPapers
      : project.total_papers_found;

  let progress = 0;
  let progressText = "Initializing...";
  let isIndeterminate = false;

  const BASE_SEARCHING = 5;
  const BASE_ANALYZING = 15;
  const BASE_SYNTHESIZING = 95;

  switch (project.status) {
    case "planning":
      progress = 2;
      progressText = "Initializing workflow...";
      isIndeterminate = true;
      break;
    case "searching":
      progress = BASE_SEARCHING;
      progressText = "Searching for relevant papers...";
      isIndeterminate = true;
      break;
    case "analyzing":
      if (totalPapersToAnalyze > 0) {
        progress =
          BASE_ANALYZING +
          (papersAnalyzed / totalPapersToAnalyze) *
            (BASE_SYNTHESIZING - BASE_ANALYZING);
        progressText = `Analyzing: ${papersAnalyzed} of ${totalPapersToAnalyze} papers`;
      } else {
        progress = BASE_ANALYZING;
        progressText = "Preparing to analyze papers...";
        isIndeterminate = true;
      }
      break;
    case "synthesizing":
      progress = BASE_SYNTHESIZING;
      progressText = "Synthesizing final report...";
      isIndeterminate = true;
      break;
    default:
      return null;
  }

  return (
    <Box className="w-full mt-3">
      <Box className="flex justify-between items-center mb-1">
        <Typography variant="caption" className="text-slate-500 font-medium">
          {progressText}
        </Typography>
        {!isIndeterminate && (
          <Typography variant="caption" className="text-slate-400">
            {Math.round(progress)}%
          </Typography>
        )}
      </Box>
      <LinearProgress
        variant={isIndeterminate ? "indeterminate" : "determinate"}
        value={progress}
        sx={{
          height: 6,
          borderRadius: 3,
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          "& .MuiLinearProgress-bar": {
            borderRadius: 3,
            background: "linear-gradient(90deg, #3b82f6 0%, #14b8a6 100%)",
          },
        }}
      />
    </Box>
  );
};

const ProjectCard: React.FC<ProjectCardProps> = ({ project }) => {
  const { updateProjectStatus, removeProject } = useProjectStore();
  const isProcessing = [
    "planning",
    "searching",
    "analyzing",
    "synthesizing",
  ].includes(project.status);

  // Use WebSocket for real-time updates instead of polling
  // The hook automatically updates the project store with real-time data
  const {
    papersAnalyzed: realtimePapersAnalyzed,
    totalPapers: realtimeTotalPapers,
  } = useProjectStream(isProcessing ? project.id : undefined, {
    autoReconnect: true,
  });

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const navigate = useNavigate();

  const handleStartReview = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      toast.info(`Starting literature review for "${project.title}"...`, {
        toastId: `start-review-${project.id}`,
      });
      updateProjectStatus(project.id, "planning");
      await startLiteratureReview(project.id);
    } catch (error) {
      console.error("Failed to start literature review:", error);
      toast.error("Failed to start literature review. Please try again.");
      updateProjectStatus(project.id, "error");
    }
  };

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    setIsDeleting(true);
    await removeProject(project.id);
    setIsDeleting(false);
    setDeleteDialogOpen(false);
  };

  const handleDeleteCancel = () => {
    setDeleteDialogOpen(false);
  };

  const handleCardClick = () => {
    if (project.status === "completed") {
      navigate(`/project/${project.id}`);
    }
  };

  const isCreating = project.status === "creating";
  const isReady = project.status === "created";
  const isFailed =
    project.status === "error" || project.status === "error_no_papers_found";
  const isCompleted = project.status === "completed";

  // Format date
  const formatDate = (dateString: string) => {
    if (!dateString) return "";
    const date = new Date(dateString);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
  };

  return (
    <>
      <Card
        onClick={handleCardClick}
        sx={{
          display: "flex",
          flexDirection: "column",
          height: "100%",
          borderRadius: 3,
          border: "1px solid",
          borderColor: "rgba(0,0,0,0.08)",
          boxShadow: "0 4px 20px rgba(0,0,0,0.05)",
          transition: "all 0.3s ease",
          cursor: isCompleted ? "pointer" : "default",
          background: "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
          "&:hover": {
            transform: isCompleted ? "translateY(-4px)" : "none",
            boxShadow: isCompleted
              ? "0 12px 40px rgba(59, 130, 246, 0.15)"
              : "0 4px 20px rgba(0,0,0,0.05)",
            borderColor: isCompleted
              ? "rgba(59, 130, 246, 0.3)"
              : "rgba(0,0,0,0.08)",
          },
        }}
      >
        {/* Header Section */}
        <Box
          sx={{
            p: 2.5,
            pb: 2,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 1,
          }}
        >
          <StatusChip status={project.status} />
          <Tooltip title="Delete project" arrow>
            <IconButton
              size="small"
              onClick={handleDeleteClick}
              disabled={isDeleting || isProcessing}
              sx={{
                color: "grey.400",
                "&:hover": {
                  color: "error.main",
                  backgroundColor: "error.lighter",
                },
                "&.Mui-disabled": {
                  color: "grey.300",
                },
              }}
            >
              <DeleteOutlineIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Content Section */}
        <CardContent sx={{ flexGrow: 1, pt: 0, px: 2.5, pb: 2 }}>
          <Typography
            variant="h6"
            component="h3"
            sx={{
              fontWeight: 700,
              fontSize: "1.1rem",
              color: "slate.800",
              mb: 1.5,
              lineHeight: 1.3,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            {project.title}
          </Typography>

          <Typography
            variant="body2"
            sx={{
              color: "text.secondary",
              fontStyle: "italic",
              lineHeight: 1.5,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
              mb: 2,
            }}
          >
            "{project.research_question}"
          </Typography>

          {/* Meta Info */}
          <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
            {project.created_at && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <CalendarTodayOutlinedIcon
                  sx={{ fontSize: 14, color: "grey.400" }}
                />
                <Typography variant="caption" color="text.secondary">
                  {formatDate(project.created_at)}
                </Typography>
              </Box>
            )}
            {project.total_papers_found > 0 && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                <ArticleOutlinedIcon sx={{ fontSize: 14, color: "grey.400" }} />
                <Typography variant="caption" color="text.secondary">
                  {project.total_papers_found} papers
                </Typography>
              </Box>
            )}
          </Box>

          {/* Progress Section (only during processing) */}
          {isProcessing && (
            <ProgressTracker
              project={project}
              realtimePapersAnalyzed={realtimePapersAnalyzed}
              realtimeTotalPapers={realtimeTotalPapers}
            />
          )}
        </CardContent>

        <Divider />

        {/* Action Section */}
        <Box sx={{ p: 2, backgroundColor: "rgba(248, 250, 252, 0.5)" }}>
          {isCreating && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 1.5,
                py: 0.5,
              }}
            >
              <CircularProgress size={18} thickness={4} />
              <Typography variant="body2" color="text.secondary">
                Setting up project...
              </Typography>
            </Box>
          )}

          {isReady && (
            <Button
              fullWidth
              variant="contained"
              onClick={handleStartReview}
              startIcon={<AutoAwesomeIcon />}
              sx={{
                background: "linear-gradient(135deg, #3b82f6 0%, #14b8a6 100%)",
                textTransform: "none",
                fontWeight: 600,
                py: 1,
                borderRadius: 2,
                "&:hover": {
                  background:
                    "linear-gradient(135deg, #2563eb 0%, #0d9488 100%)",
                },
              }}
            >
              Start Literature Review
            </Button>
          )}

          {isProcessing && (
            <Chip
              label="Processing..."
              size="small"
              sx={{
                width: "100%",
                backgroundColor: "rgba(59, 130, 246, 0.1)",
                color: "primary.main",
                fontWeight: 500,
              }}
            />
          )}

          {isCompleted && (
            <Button
              component={Link}
              to={`/project/${project.id}`}
              fullWidth
              variant="outlined"
              endIcon={<ArrowForwardIcon />}
              onClick={(e) => e.stopPropagation()}
              sx={{
                textTransform: "none",
                fontWeight: 600,
                py: 1,
                borderRadius: 2,
                borderColor: "primary.main",
                color: "primary.main",
                "&:hover": {
                  backgroundColor: "primary.main",
                  color: "white",
                  borderColor: "primary.main",
                },
              }}
            >
              View Results
            </Button>
          )}

          {isFailed && (
            <Button
              fullWidth
              variant="contained"
              color="warning"
              onClick={handleStartReview}
              startIcon={<RefreshIcon />}
              sx={{
                textTransform: "none",
                fontWeight: 600,
                py: 1,
                borderRadius: 2,
              }}
            >
              Retry Review
            </Button>
          )}
        </Box>
      </Card>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={handleDeleteCancel}
        PaperProps={{
          sx: {
            borderRadius: 3,
            maxWidth: 400,
            backgroundColor: "#ffffff",
            backgroundImage: "none",
          },
        }}
      >
        <DialogTitle sx={{ pb: 1, fontWeight: 600, color: "#1e293b" }}>
          Delete Project?
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ color: "#64748b" }}>
            Are you sure you want to delete <strong>"{project.title}"</strong>?
            This will permanently remove the project and all its associated data
            including paper references and analysis results.
          </DialogContentText>
          <Box
            sx={{
              mt: 2,
              p: 1.5,
              borderRadius: 2,
              backgroundColor: "#fee2e2",
              border: "1px solid #fecaca",
            }}
          >
            <Typography
              variant="caption"
              sx={{ color: "#dc2626", fontWeight: 500 }}
            >
              ⚠️ This action cannot be undone.
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions sx={{ p: 2, pt: 1 }}>
          <Button
            onClick={handleDeleteCancel}
            disabled={isDeleting}
            sx={{ textTransform: "none" }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
            disabled={isDeleting}
            startIcon={
              isDeleting ? (
                <CircularProgress size={16} color="inherit" />
              ) : (
                <DeleteOutlineIcon />
              )
            }
            sx={{
              textTransform: "none",
              borderRadius: 2,
            }}
          >
            {isDeleting ? "Deleting..." : "Delete Project"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ProjectCard;
