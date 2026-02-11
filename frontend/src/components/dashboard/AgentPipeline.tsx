import React from "react";
import { Box, Typography, LinearProgress, Chip, Paper, styled, keyframes } from "@mui/material";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked";
import ChangeCircleIcon from "@mui/icons-material/ChangeCircle";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import HubIcon from "@mui/icons-material/Hub";


const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`;

const pulse = keyframes`
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.9); }
`;


const PipelineContainer = styled(Paper)(() => ({
  padding: "24px",
  marginBottom: "24px",
  backgroundColor: "rgba(24, 24, 27, 0.7)", 
  backdropFilter: "blur(12px)",
  border: "1px solid #27272F",
  borderRadius: "16px",
  color: "#F4F4F5",
  position: "relative",
  overflow: "hidden",
  "&::before": {
    content: '""',
    position: "absolute",
    top: 0,
    left: 0,
    width: "4px",
    height: "100%",
    background: "linear-gradient(to bottom, #FFB900, #00B894)",
  }
}));

const AgentStepBox = styled(Box, {
  shouldForwardProp: (prop) => prop !== "isActive" && prop !== "isCompleted",
})<{ isActive?: boolean; isCompleted?: boolean }>(({ isActive }) => ({
  display: "flex",
  alignItems: "center",
  padding: "16px",
  borderRadius: "12px",
  transition: "all 0.3s ease",
  backgroundColor: isActive ? "rgba(255, 185, 0, 0.08)" : "transparent",
  border: isActive ? "1px solid rgba(255, 185, 0, 0.3)" : "1px solid transparent",
  position: "relative",
  zIndex: 1,
}));

const ConnectorLine = styled(Box, {
  shouldForwardProp: (prop) => prop !== "isCompleted",
})<{ isCompleted?: boolean }>(({ isCompleted }) => ({
  width: "2px",
  height: "20px",
  marginLeft: "27px",
  backgroundColor: isCompleted ? "#00B894" : "#27272F",
  transition: "background-color 0.5s ease",
}));

const TerminalBox = styled(Box)(() => ({
  marginTop: "24px",
  backgroundColor: "#09090B",
  border: "1px solid #27272F",
  borderRadius: "8px",
  padding: "16px",
  fontFamily: "'JetBrains Mono', monospace",
  fontSize: "0.8rem",
  color: "#A1A1AA",
  maxHeight: "180px",
  overflowY: "auto",
  "&::-webkit-scrollbar": { width: "4px" },
  "&::-webkit-scrollbar-thumb": { backgroundColor: "#3F3F46", borderRadius: "2px" },
}));

const AGENTS = [
  { id: "planner", name: "Research Planner", desc: "Constructing heuristic search map" },
  { id: "retriever", name: "Source Retriever", desc: "Acquiring cross-domain literature" },
  { id: "analyzer", name: "Insight Analyzer", desc: "Extracting semantic relationships" },
  { id: "synthesizer", name: "Synthesizer", desc: "Drafting multidimensional summary" },
];

interface AgentPipelineProps {
  currentAgent: string | null;
  progress: number;
  logs: string[];
  isConnected?: boolean;
  projectStatus?: string;
}

export function AgentPipeline({
  currentAgent,
  progress,
  logs,
  isConnected = false,
  projectStatus,
}: AgentPipelineProps) {
  
  const getAgentStatus = (agentId: string) => {
    if (projectStatus === "completed") return "completed";
    if (projectStatus?.startsWith("error")) {
        const currentIdx = AGENTS.findIndex(a => a.id === currentAgent);
        const agentIdx = AGENTS.findIndex(a => a.id === agentId);
        if (agentIdx === currentIdx) return "error";
        return agentIdx < currentIdx ? "completed" : "pending";
    }
    const currentIdx = AGENTS.findIndex(a => a.id === currentAgent);
    const agentIdx = AGENTS.findIndex(a => a.id === agentId);
    if (currentIdx === -1) return "pending";
    if (agentIdx < currentIdx) return "completed";
    return agentIdx === currentIdx ? "active" : "pending";
  };

  const renderIcon = (status: string) => {
    switch (status) {
      case "completed": return <CheckCircleIcon sx={{ color: "#00B894" }} />;
      case "active": return <ChangeCircleIcon sx={{ color: "#FFB900", animation: `${spin} 3s linear infinite` }} />;
      case "error": return <ErrorOutlineIcon sx={{ color: "#EF4444" }} />;
      default: return <RadioButtonUncheckedIcon sx={{ color: "#3F3F46" }} />;
    }
  };

  return (
    <PipelineContainer elevation={0}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <HubIcon sx={{ color: "#FFB900" }} />
          <Typography variant="h6" sx={{ fontWeight: 800, letterSpacing: "-0.02em", color: "#F4F4F5" }}>
            Neural Pipeline
          </Typography>
        </Box>
        <Chip
          label={isConnected ? "ULTRALINK ACTIVE" : "OFFLINE"}
          sx={{
            height: 24,
            fontSize: "0.65rem",
            fontWeight: 900,
            backgroundColor: isConnected ? "rgba(0, 184, 148, 0.1)" : "rgba(63, 63, 70, 0.1)",
            color: isConnected ? "#00B894" : "#71717A",
            border: `1px solid ${isConnected ? "#00B894" : "#3F3F46"}`,
            "& .MuiChip-icon": {
                animation: isConnected ? `${pulse} 2s infinite` : "none",
                color: "inherit"
            }
          }}
          icon={<Box component="span" sx={{ width: 6, height: 6, borderRadius: "50%", bgcolor: "currentColor", ml: 1 }} />}
        />
      </Box>

      {/* Global Progress Bar */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
          <Typography variant="caption" sx={{ color: "#71717A", fontWeight: 700, textTransform: "uppercase" }}>
            Synthesis Progress
          </Typography>
          <Typography variant="caption" sx={{ color: "#FFB900", fontWeight: 800 }}>
            {Math.round(progress)}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 6,
            borderRadius: 3,
            backgroundColor: "#18181B",
            "& .MuiLinearProgress-bar": {
              borderRadius: 3,
              background: "linear-gradient(90deg, #FFB900 0%, #00B894 100%)",
            },
          }}
        />
      </Box>

      {/* Agent Steps */}
      <Box>
        {AGENTS.map((agent, index) => {
          const status = getAgentStatus(agent.id);
          const isLast = index === AGENTS.length - 1;
          return (
            <React.Fragment key={agent.id}>
              <AgentStepBox isActive={status === "active"} isCompleted={status === "completed"}>
                <Box sx={{ mr: 2.5, display: "flex" }}>{renderIcon(status)}</Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle2" sx={{ 
                    color: status === "pending" ? "#52525B" : "#F4F4F5",
                    fontWeight: status === "active" ? 800 : 600
                  }}>
                    {agent.name}
                  </Typography>
                  <Typography variant="caption" sx={{ color: "#71717A", display: "block", mt: -0.5 }}>
                    {agent.desc}
                  </Typography>
                </Box>
                {status === "active" && (
                   <Typography variant="caption" sx={{ color: "#FFB900", fontWeight: 900, fontSize: '0.6rem' }}>
                    PROCESSING...
                   </Typography>
                )}
              </AgentStepBox>
              {!isLast && <ConnectorLine isCompleted={status === "completed"} />}
            </React.Fragment>
          );
        })}
      </Box>

      {/* Terminal Activity Log */}
      {logs.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" sx={{ color: "#71717A", fontWeight: 700, ml: 1, textTransform: "uppercase" }}>
            Agent Telemetry
          </Typography>
          <TerminalBox>
            {logs.slice(-12).map((log, i) => (
              <Box key={i} sx={{ mb: 0.5, opacity: 0.4 + (i / 12) * 0.6, display: "flex", gap: 1 }}>
                <Typography component="span" sx={{ color: "#FFB900", fontSize: "inherit", fontWeight: 700 }}>
                  &gt;
                </Typography>
                <Typography component="span" sx={{ fontSize: "inherit" }}>
                  {log}
                </Typography>
              </Box>
            ))}
          </TerminalBox>
        </Box>
      )}
    </PipelineContainer>
  );
}

export default AgentPipeline;