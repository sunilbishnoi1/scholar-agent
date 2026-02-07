import { useState, useEffect } from "react";
import {
  Button,
  Container,
  Typography,
  Box,
  CircularProgress,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import ProjectCard from "../components/dashboard/ProjectCard";
import CreateProjectModal from "../components/dashboard/CreateProjectModal";
import { useProjectStore } from "../store/projectStore";
import { useAuthStore } from "../store/authStore";
import type { ResearchProject } from "../types";

const DashboardPage = () => {
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
    <Container maxWidth="lg" className="mt-10 pt-10 sm:pt-12">
      <Box className="flex flex-row md:flex-row justify-between items-start md:items-center mb-4">
        <Typography
          variant="h4"
          component="h1"
          className="bg-gradient-to-r from-teal-600 to-teal-500 bg-clip-text text-transparent font-bold mb-4 md:mb-0"
        >
          My Research Projects
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setIsModalOpen(true)}
          className="bg-gradient-to-r from-blue-600 to-teal-500 hover:bg-blue-700 text-white"
        >
          <span className="block md:hidden">New</span>
          <span className="hidden md:inline">New Project</span>
        </Button>
      </Box>

      {isLoading && <CircularProgress className="block mx-auto" />}

      {!isLoading && projects.length === 0 ? (
        <Box className="text-center my-16">
          <Typography variant="h6" className="text-slate-500">
            No projects yet.
          </Typography>
          <Typography className="text-slate-500">
            Click "New Project" to get started!
          </Typography>
        </Box>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-8">
          {projects.map((project: ResearchProject) => (
            <ProjectCard key={project.id} project={project} />
          ))}
        </div>
      )}

      <CreateProjectModal
        open={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </Container>
  );
};

export default DashboardPage;
