import { Box, Typography, LinearProgress, Chip, Paper } from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import LoopIcon from "@mui/icons-material/Loop";
import ErrorIcon from "@mui/icons-material/Error";

interface AgentPipelineProps {
  /** Currently active agent */
  currentAgent: string | null;
  /** Overall progress percentage (0-100) */
  progress: number;
  /** Recent log messages to display */
  logs: string[];
  /** Whether WebSocket is connected */
  isConnected?: boolean;
  /** Project status for determining completed state */
  projectStatus?: string;
}

interface AgentConfig {
  id: string;
  name: string;
  description: string;
}

const AGENTS: AgentConfig[] = [
  {
    id: "planner",
    name: "Research Planner",
    description: "Creating search strategy",
  },
  {
    id: "retriever",
    name: "Paper Retriever",
    description: "Fetching academic papers",
  },
  {
    id: "analyzer",
    name: "Paper Analyzer",
    description: "Analyzing relevance & insights",
  },
  {
    id: "synthesizer",
    name: "Synthesizer",
    description: "Writing literature review",
  },
];

type AgentStatus = "pending" | "active" | "completed" | "error";

/**
 * Visualizes the agent pipeline with real-time progress tracking.
 *
 * Shows each agent's status (pending, active, completed) and displays
 * a live activity log from WebSocket updates.
 */
export function AgentPipeline({
  currentAgent,
  progress,
  logs,
  isConnected = false,
  projectStatus,
}: AgentPipelineProps) {
  const getAgentStatus = (agentId: string): AgentStatus => {
    // If project is completed, all agents are completed
    if (projectStatus === "completed") {
      return "completed";
    }

    // If project has error, show error on current agent
    if (projectStatus?.startsWith("error")) {
      const currentIndex = AGENTS.findIndex((a) => a.id === currentAgent);
      const agentIndex = AGENTS.findIndex((a) => a.id === agentId);
      if (agentIndex === currentIndex) return "error";
      if (agentIndex < currentIndex) return "completed";
      return "pending";
    }

    const agentIndex = AGENTS.findIndex((a) => a.id === agentId);
    const currentIndex = AGENTS.findIndex((a) => a.id === currentAgent);

    if (currentIndex === -1) return "pending";
    if (agentIndex < currentIndex) return "completed";
    if (agentIndex === currentIndex) return "active";
    return "pending";
  };

  const getStatusIcon = (status: AgentStatus) => {
    switch (status) {
      case "completed":
        return <CheckCircleIcon sx={{ color: "success.main" }} />;
      case "active":
        return (
          <LoopIcon
            sx={{
              color: "primary.main",
              animation: "spin 1s linear infinite",
              "@keyframes spin": {
                "0%": { transform: "rotate(0deg)" },
                "100%": { transform: "rotate(360deg)" },
              },
            }}
          />
        );
      case "error":
        return <ErrorIcon sx={{ color: "error.main" }} />;
      default:
        return <RadioButtonUncheckedIcon sx={{ color: "grey.400" }} />;
    }
  };

  const getStatusColor = (
    status: AgentStatus,
  ): "success" | "primary" | "error" | "default" => {
    switch (status) {
      case "completed":
        return "success";
      case "active":
        return "primary";
      case "error":
        return "error";
      default:
        return "default";
    }
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      {/* Header with connection status */}
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          mb: 2,
        }}
      >
        <Typography variant="h6" fontWeight="semibold">
          Agent Pipeline
        </Typography>
        <Chip
          size="small"
          label={isConnected ? "Live" : "Offline"}
          color={isConnected ? "success" : "default"}
          variant="outlined"
          sx={{
            "& .MuiChip-label": {
              display: "flex",
              alignItems: "center",
              gap: 0.5,
            },
          }}
          icon={
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                bgcolor: isConnected ? "success.main" : "grey.400",
                animation: isConnected ? "pulse 2s infinite" : "none",
                "@keyframes pulse": {
                  "0%, 100%": { opacity: 1 },
                  "50%": { opacity: 0.5 },
                },
              }}
            />
          }
        />
      </Box>

      {/* Progress Bar */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Overall Progress
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {Math.round(progress)}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 8,
            borderRadius: 4,
            bgcolor: "grey.200",
            "& .MuiLinearProgress-bar": {
              borderRadius: 4,
            },
          }}
        />
      </Box>

      {/* Agent Steps */}
      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {AGENTS.map((agent, index) => {
          const status = getAgentStatus(agent.id);
          return (
            <Box key={agent.id}>
              {/* Connector line */}
              {index > 0 && (
                <Box
                  sx={{
                    position: "relative",
                    ml: 1.5,
                    mt: -2,
                    mb: -1,
                    width: 2,
                    height: 16,
                    bgcolor:
                      getAgentStatus(AGENTS[index - 1].id) === "completed"
                        ? "success.main"
                        : "grey.300",
                  }}
                />
              )}

              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  p: 1.5,
                  borderRadius: 2,
                  bgcolor: status === "active" ? "primary.50" : "transparent",
                  transition: "background-color 0.2s",
                }}
              >
                {/* Status Icon */}
                <Box sx={{ mr: 2 }}>{getStatusIcon(status)}</Box>

                {/* Agent Info */}
                <Box sx={{ flex: 1 }}>
                  <Typography
                    variant="subtitle2"
                    fontWeight={status === "active" ? "semibold" : "normal"}
                  >
                    {agent.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {agent.description}
                  </Typography>
                </Box>

                {/* Status Chip */}
                <Chip
                  size="small"
                  label={status}
                  color={getStatusColor(status)}
                  variant={status === "active" ? "filled" : "outlined"}
                />
              </Box>
            </Box>
          );
        })}
      </Box>

      {/* Live Logs */}
      {logs.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Live Activity
          </Typography>
          <Box
            sx={{
              bgcolor: "grey.900",
              color: "success.light",
              p: 2,
              borderRadius: 2,
              fontFamily: "monospace",
              fontSize: "0.75rem",
              maxHeight: 160,
              overflow: "auto",
              "&::-webkit-scrollbar": {
                width: 6,
              },
              "&::-webkit-scrollbar-thumb": {
                bgcolor: "grey.700",
                borderRadius: 3,
              },
            }}
          >
            {logs.slice(-10).map((log, i) => (
              <Box key={i} sx={{ mb: 0.5, opacity: 0.7 + (i / 10) * 0.3 }}>
                <Typography
                  component="span"
                  sx={{ color: "grey.500", mr: 1, fontSize: "inherit" }}
                >
                  [{currentAgent || "system"}]
                </Typography>
                {log}
              </Box>
            ))}
          </Box>
        </Box>
      )}
    </Paper>
  );
}

export default AgentPipeline;
