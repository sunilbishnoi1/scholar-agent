import React, { useState, useEffect } from "react";
import {
  Button,
  Container,
  Typography,
  Box,
  CircularProgress,
  Paper,
} from "@mui/material";
import { styled } from "@mui/system";
import AddIcon from "@mui/icons-material/Add";
import SchoolIcon from "@mui/icons-material/School";
import ProjectCard from "../components/dashboard/ProjectCard";
import CreateProjectModal from "../components/dashboard/CreateProjectModal";
import { useProjectStore } from "../store/projectStore";
import { useAuthStore } from "../store/authStore";
import type { ResearchProject } from "../types";

const PageWrapper = styled(Box)({
  minHeight: "100vh",
  backgroundColor: "#09090B", 
  color: "#F4F4F5",
  paddingTop: "100px", 
  paddingBottom: "40px",
  backgroundImage:
    "radial-gradient(circle at 50% -20%, rgba(255, 185, 0, 0.05) 0%, transparent 50%)",
});

const HeaderSection = styled(Box)(() => ({
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "3rem",
  gap: "1rem",
}));

const NoirButton = styled(Button)({
  backgroundColor: "#FFB900",
  color: "#09090B",
  fontWeight: 700,
  textTransform: "none",
  padding: "10px 24px",
  borderRadius: "8px",
  fontSize: "0.95rem",
  boxShadow: "0 4px 14px 0 rgba(255, 185, 0, 0.3)",
  transition: "all 0.2s ease",
  "&:hover": {
    backgroundColor: "#E6A600",
    transform: "translateY(-2px)",
    boxShadow: "0 6px 20px rgba(255, 185, 0, 0.4)",
  },
});

const EmptyStateCard = styled(Paper)({
  backgroundColor: "rgba(24, 24, 27, 0.4)",
  backdropFilter: "blur(12px)",
  border: "1px dashed #27272F",
  borderRadius: "16px",
  padding: "4rem 2rem",
  textAlign: "center",
  color: "#A1A1AA",
});

const DashboardPage: React.FC = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const { projects, isLoading, fetchProjects } = useProjectStore();
  const user = useAuthStore((s) => s.user);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  useEffect(() => {
    if (user && isAuthenticated) {
      fetchProjects();
    }
  }, [fetchProjects, user, isAuthenticated]);

  return (
    <PageWrapper>
      <Container maxWidth="lg">
        <HeaderSection>
          <Box>
            <Typography
              variant="h4"
              component="h1"
              sx={{
                fontWeight: 800,
                letterSpacing: "-0.03em",
                background: "linear-gradient(135deg, #F4F4F5 0%, #A1A1AA 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                mb: 0.5,
                fontSize: { xs: "1.25rem", sm: "1.6rem", md: "2rem" },
                lineHeight: 1.05,
              }}
            >
              Research Workspace
            </Typography>
            <Typography
              variant="body1"
              sx={{
                color: "#71717A",
                fontFamily: "'Crimson Pro', serif",
                fontSize: "1.1rem",
              }}
            >
              Welcome back, {user?.name?.split(" ")[0] || "Researcher"}.
            </Typography>
          </Box>

          <NoirButton
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setIsModalOpen(true)}
            sx={{
              minWidth: { xs: 88, sm: 140 },
              px: { xs: 1.5, sm: 3 },
            }}
          >
            <Box
              component="span"
              sx={{ display: { xs: "inline", sm: "none" } }}
            >
              New
            </Box>
            <Box
              component="span"
              sx={{ display: { xs: "none", sm: "inline" } }}
            >
              New Project
            </Box>
          </NoirButton>
        </HeaderSection>

        {/* Projects Content */}
        {isLoading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 10 }}>
            <CircularProgress sx={{ color: "#FFB900" }} />
          </Box>
        ) : projects.length === 0 ? (
          <EmptyStateCard elevation={0}>
            <SchoolIcon sx={{ fontSize: 48, mb: 2, opacity: 0.2 }} />
            <Typography variant="h6" sx={{ color: "#F4F4F5", mb: 1 }}>
              The library is empty
            </Typography>
            <Typography
              variant="body2"
              sx={{ maxWidth: 300, mx: "auto", mb: 3 }}
            >
              You haven't initiated any research projects yet. Start your first
              deep-dive now.
            </Typography>
            <Button
              onClick={() => setIsModalOpen(true)}
              sx={{ color: "#00F5C8", fontWeight: 600, textTransform: "none" }}
            >
              + Create your first project
            </Button>
          </EmptyStateCard>
        ) : (
          <Box
            component="div"
            sx={{
              display: "grid",
              gap: 4,
              gridTemplateColumns: {
                xs: "1fr",
                md: "repeat(2, 1fr)",
              },
            }}
          >
            {projects.map((project: ResearchProject) => (
              <Box key={project.id}>
                <ProjectCard project={project} />
              </Box>
            ))}
          </Box>
        )}
      </Container>

      {/* Modal */}
      <CreateProjectModal
        open={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </PageWrapper>
  );
};

export default DashboardPage;
