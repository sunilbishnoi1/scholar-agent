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
  Divider,
} from "@mui/material";
import { styled } from "@mui/system";
import { useNavigate } from "react-router-dom";
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

const NoirCard = styled(Card)<{ isCompleted: boolean }>(({ isCompleted }) => ({
  display: "flex",
  flexDirection: "column",
  height: "100%",
  borderRadius: "16px",
  backgroundColor: "rgba(24, 24, 27, 0.6)",
  backdropFilter: "blur(12px)",
  border: "1px solid #27272F",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  cursor: isCompleted ? "pointer" : "default",
  position: "relative",
  overflow: "hidden",
  "&:hover": {
    transform: isCompleted ? "translateY(-6px)" : "none",
    borderColor: isCompleted ? "#FFB900" : "#3F3F46",
    boxShadow: isCompleted
      ? "0 12px 30px rgba(0,0,0,0.5), 0 0 20px rgba(255, 185, 0, 0.1)"
      : "none",
    "& .card-glow": { opacity: 1 },
  },
}));

const CardGlow = styled(Box)({
  position: "absolute",
  top: 0,
  left: 0,
  width: "100%",
  height: "100%",
  background:
    "radial-gradient(circle at 50% 0%, rgba(255, 185, 0, 0.08), transparent 70%)",
  opacity: 0,
  transition: "opacity 0.4s ease",
  pointerEvents: "none",
});

const ProgressText = styled(Typography)({
  fontFamily: "'Inter', sans-serif",
  fontSize: "0.75rem",
  color: "#71717A",
  fontWeight: 500,
});

const ActionButton = styled(Button)<{ variant_type?: "amber" | "outline" }>(
  ({ variant_type }) => ({
    textTransform: "none",
    fontWeight: 700,
    borderRadius: "8px",
    padding: "8px 16px",
    transition: "all 0.2s ease",
    ...(variant_type === "amber"
      ? {
          backgroundColor: "#FFB900",
          color: "#09090B",
          "&:hover": { backgroundColor: "#E6A600" },
        }
      : {
          borderColor: "#27272F",
          color: "#F4F4F5",
          "&:hover": {
            borderColor: "#FFB900",
            backgroundColor: "rgba(255, 185, 0, 0.05)",
          },
        }),
  }),
);

const ProgressTracker: React.FC<{
  project: ResearchProject;
  realtimePapersAnalyzed?: number;
  realtimeTotalPapers?: number;
}> = ({ project, realtimePapersAnalyzed, realtimeTotalPapers }) => {
  const papersAnalyzed =
    realtimePapersAnalyzed ??
    project.agent_plans.filter((p) => p.agent_type === "analyzer").length;
  const totalPapersToAnalyze =
    realtimeTotalPapers && realtimeTotalPapers > 0
      ? realtimeTotalPapers
      : project.total_papers_found;

  let progress = 0;
  let progressText = "Initializing...";
  let isIndeterminate = false;

  switch (project.status) {
    case "planning":
      progress = 5;
      isIndeterminate = true;
      break;
    case "searching":
      progress = 15;
      progressText = "Scouring databases...";
      isIndeterminate = true;
      break;
    case "analyzing":
      if (totalPapersToAnalyze > 0) {
        progress = 20 + (papersAnalyzed / totalPapersToAnalyze) * 70;
        progressText = `Extracting insights: ${papersAnalyzed}/${totalPapersToAnalyze}`;
      } else {
        progress = 20;
        isIndeterminate = true;
      }
      break;
    case "synthesizing":
      progress = 95;
      progressText = "Drafting final synthesis...";
      isIndeterminate = true;
      break;
    default:
      return null;
  }

  return (
    <Box sx={{ mt: 3, width: "100%" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
        <ProgressText>{progressText}</ProgressText>
        {!isIndeterminate && (
          <ProgressText sx={{ color: "#FFB900" }}>
            {Math.round(progress)}%
          </ProgressText>
        )}
      </Box>
      <LinearProgress
        variant={isIndeterminate ? "indeterminate" : "determinate"}
        value={progress}
        sx={{
          height: 4,
          borderRadius: 2,
          backgroundColor: "rgba(255, 255, 255, 0.05)",
          "& .MuiLinearProgress-bar": {
            borderRadius: 2,
            background: "linear-gradient(90deg, #FFB900 0%, #00F5C8 100%)",
          },
        }}
      />
    </Box>
  );
};

const ProjectCard: React.FC<{ project: ResearchProject }> = ({ project }) => {
  const { updateProjectStatus, removeProject } = useProjectStore();
  const isProcessing = [
    "planning",
    "searching",
    "analyzing",
    "synthesizing",
  ].includes(project.status);
  const { papersAnalyzed, totalPapers } = useProjectStream(
    isProcessing ? project.id : undefined,
    { autoReconnect: true },
  );

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const navigate = useNavigate();

  const handleStartReview = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      updateProjectStatus(project.id, "planning");
      await startLiteratureReview(project.id);
    } catch (err) {
      console.error(err);
      toast.error("Deployment failed. Re-initialize system.");
      updateProjectStatus(project.id, "error");
    }
  };

  const isCompleted = project.status === "completed";
  const isFailed =
    project.status === "error" || project.status === "error_no_papers_found";

  return (
    <>
      <NoirCard
        isCompleted={isCompleted}
        onClick={() => isCompleted && navigate(`/project/${project.id}`)}
      >
        <CardGlow className="card-glow" />

        <Box
          sx={{
            p: 2.5,
            pb: 0,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            zIndex: 1,
          }}
        >
          <StatusChip status={project.status} />
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              setDeleteDialogOpen(true);
            }}
            sx={{ color: "#3F3F46", "&:hover": { color: "#EF4444" } }}
          >
            <DeleteOutlineIcon fontSize="small" />
          </IconButton>
        </Box>

        <CardContent sx={{ flexGrow: 1, px: 2.5, py: 2.5, zIndex: 1 }}>
          <Typography
            variant="h6"
            sx={{
              fontWeight: 700,
              color: "#F4F4F5",
              mb: 1,
              letterSpacing: "-0.01em",
              lineHeight: 1.3,
            }}
          >
            {project.title}
          </Typography>

          <Typography
            variant="body2"
            sx={{
              color: "#A1A1AA",
              fontFamily: "'Crimson Pro', serif",
              fontStyle: "italic",
              fontSize: "1.05rem",
              mb: 2,
              display: "-webkit-box",
              WebkitLineClamp: 2,
              WebkitBoxOrient: "vertical",
              overflow: "hidden",
            }}
          >
            "{project.research_question}"
          </Typography>

          <Box sx={{ display: "flex", gap: 2.5, mt: "auto" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.7 }}>
              <CalendarTodayOutlinedIcon
                sx={{ fontSize: 14, color: "#52525B" }}
              />
              <Typography variant="caption" sx={{ color: "#71717A" }}>
                {new Date(project.created_at).toLocaleDateString()}
              </Typography>
            </Box>
            {project.total_papers_found > 0 && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.7 }}>
                <ArticleOutlinedIcon sx={{ fontSize: 14, color: "#52525B" }} />
                <Typography variant="caption" sx={{ color: "#71717A" }}>
                  {project.total_papers_found} Sources
                </Typography>
              </Box>
            )}
          </Box>

          {isProcessing && (
            <ProgressTracker
              project={project}
              realtimePapersAnalyzed={papersAnalyzed}
              realtimeTotalPapers={totalPapers}
            />
          )}
        </CardContent>

        <Divider sx={{ borderColor: "#27272F", opacity: 0.5 }} />

        <Box sx={{ p: 2, backgroundColor: "rgba(9, 9, 11, 0.4)", zIndex: 1 }}>
          {project.status === "created" && (
            <ActionButton
              variant_type="amber"
              fullWidth
              onClick={handleStartReview}
              startIcon={<AutoAwesomeIcon />}
            >
              Initiate Agents
            </ActionButton>
          )}

          {isCompleted && (
            <ActionButton
              variant_type="outline"
              variant="outlined"
              fullWidth
              endIcon={<ArrowForwardIcon />}
            >
              Examine Findings
            </ActionButton>
          )}

          {isFailed && (
            <ActionButton
              variant_type="amber"
              fullWidth
              onClick={handleStartReview}
              startIcon={<RefreshIcon />}
            >
              Retry Deployment
            </ActionButton>
          )}

          {isProcessing && (
            <Box sx={{ textAlign: "center", py: 0.5 }}>
              <Typography
                variant="caption"
                sx={{
                  color: "#00F5C8",
                  fontWeight: 700,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}
              >
                Neural Link Active
              </Typography>
            </Box>
          )}
        </Box>
      </NoirCard>

      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        PaperProps={{
          sx: {
            backgroundColor: "#18181B",
            border: "1px solid #27272F",
            borderRadius: "12px",
            color: "#F4F4F5",
          },
        }}
      >
        <DialogTitle sx={{ fontWeight: 800 }}>Terminate Project?</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ color: "#A1A1AA" }}>
            Deconstructing this workspace will permanently erase all synthesized
            insights for <strong>"{project.title}"</strong>.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ p: 2.5 }}>
          <Button
            onClick={() => setDeleteDialogOpen(false)}
            sx={{ color: "#71717A", textTransform: "none" }}
          >
            Abort
          </Button>
          <Button
            onClick={async () => {
              setIsDeleting(true);
              await removeProject(project.id);
            }}
            variant="contained"
            color="error"
            sx={{ borderRadius: "8px", textTransform: "none", fontWeight: 700 }}
          >
            {isDeleting ? (
              <CircularProgress size={20} color="inherit" />
            ) : (
              "Delete Workspace"
            )}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ProjectCard;
