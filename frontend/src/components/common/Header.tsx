import React, { useState, useEffect } from "react";
import { useNavigate, Link, useLocation } from "react-router-dom";
import {
  AccountCircle,
  Logout,
  Build as BuildIcon,
} from "@mui/icons-material";
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  Typography,
  Divider,
  Box,
} from "@mui/material";
import { useAuthStore } from "../../store/authStore";
import icon from "../../assets/SA_icon-192.png";

const Header = () => {
  const { isAuthenticated, logout, user, fetchUser } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);

  useEffect(() => {
    if (isAuthenticated && !user) {
      fetchUser();
    }
  }, [isAuthenticated, user, fetchUser]);

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleClose();
    logout();
    navigate("/login");
  };

  const handleNavigate = (path: string) => {
    handleClose();
    navigate(path);
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="fixed top-0 w-full z-50 transition-all duration-300 bg-[#09090BD9] backdrop-blur-lg border-b border-[var(--border-subtle)]">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            to={isAuthenticated ? "/dashboard" : "/"}
            className="flex items-center no-underline gap-3 group"
          >
            <div className="relative flex items-center justify-center">
              <div className="absolute inset-0 bg-[var(--accent-primary)] opacity-20 blur-md rounded-full group-hover:opacity-30 transition-opacity duration-300"></div>
              <img src={icon} className="h-8 w-8 relative z-10" alt="Scholar Agent Logo" />
            </div>
            <span 
              className="font-bold text-xl tracking-tight"
              style={{
                fontFamily: 'var(--font-primary)',
                background: 'linear-gradient(135deg, var(--color-insight-500) 0%, var(--color-aurora-500) 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Scholar Agent
            </span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-6 absolute left-1/2 transform -translate-x-1/2">
            {isAuthenticated ? (
              <Link
                to="/tools"
                className={`relative px-3 py-2 text-sm font-medium transition-colors duration-200 ${
                  isActive('/tools') 
                    ? 'text-[var(--accent-primary)]' 
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                Tools
                {isActive('/tools') && (
                  <span className="absolute bottom-0 left-0 w-full h-0.5 bg-[var(--accent-primary)] rounded-full shadow-[0_0_8px_var(--accent-primary)]" />
                )}
              </Link>
            ) : (
              <Link
                to="/how-it-works"
                className={`relative px-3 py-2 text-sm font-medium transition-colors duration-200 ${
                  isActive('/how-it-works') 
                    ? 'text-[var(--accent-primary)]' 
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
              >
                How It Works
                {isActive('/how-it-works') && (
                  <span className="absolute bottom-0 left-0 w-full h-0.5 bg-[var(--accent-primary)] rounded-full shadow-[0_0_8px_var(--accent-primary)]" />
                )}
              </Link>
            )}
          </nav>

          {/* Right Side Actions */}
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <>
                <IconButton
                  onClick={handleMenu}
                  size="small"
                  aria-controls={open ? 'account-menu' : undefined}
                  aria-haspopup="true"
                  aria-expanded={open ? 'true' : undefined}
                  className="transition-transform duration-200 hover:scale-105"
                  sx={{ 
                    padding: '4px',
                    border: '1px solid transparent',
                    '&:hover': { backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-subtle)' }
                  }}
                >
                  <AccountCircle
                    sx={{ fontSize: '2rem', color: 'var(--text-secondary)', '&:hover': { color: 'var(--accent-primary)' } }}
                  />
                </IconButton>
                <Menu
                  anchorEl={anchorEl}
                  id="account-menu"
                  open={open}
                  onClose={handleClose}
                  onClick={handleClose}
                  transformOrigin={{ horizontal: 'right', vertical: 'top' }}
                  anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
                  PaperProps={{
                    elevation: 0,
                    sx: {
                      overflow: 'visible',
                      filter: 'drop-shadow(0px 4px 20px rgba(0,0,0,0.4))',
                      mt: 1.5,
                      minWidth: 240,
                      backgroundColor: 'var(--bg-page)',
                      backgroundImage: 'linear-gradient(rgba(255,255,255,0.03), rgba(255,255,255,0.0))',
                      backdropFilter: 'blur(16px)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: '12px',
                      color: 'var(--text-primary)',
                      '& .MuiMenuItem-root': {
                        fontSize: '0.95rem',
                        fontWeight: 500,
                        padding: '10px 20px',
                        margin: '4px 8px',
                        borderRadius: '8px',
                        '&:hover': {
                          backgroundColor: 'var(--bg-elevated)',
                        },
                      },
                      '&:before': {
                        content: '""',
                        display: 'block',
                        position: 'absolute',
                        top: 0,
                        right: 14,
                        width: 10,
                        height: 10,
                        bgcolor: 'var(--bg-page)',
                        borderTop: '1px solid var(--border-subtle)',
                        borderLeft: '1px solid var(--border-subtle)',
                        transform: 'translateY(-50%) rotate(45deg)',
                        zIndex: 0,
                      },
                    },
                  }}
                >
                  <Box sx={{ px: 3, py: 2 }}>
                    <Typography variant="subtitle1" fontWeight="700" color="var(--text-primary)">
                      {user?.name || "User"}
                    </Typography>
                    <Typography variant="body2" color="var(--text-secondary)" sx={{ opacity: 0.8 }}>
                      {user?.email || ""}
                    </Typography>
                  </Box>
                  <Divider sx={{ borderColor: 'var(--border-subtle)', my: 1 }} />
                  
                  {/* Mobile Navigation inside Menu */}
                  <Box sx={{ display: { xs: "block", md: "none" } }}>
                    <MenuItem onClick={() => handleNavigate("/tools")}>
                      <ListItemIcon>
                        <BuildIcon fontSize="small" sx={{ color: 'var(--text-secondary)' }} />
                      </ListItemIcon>
                      Tools
                    </MenuItem>
                    <Divider sx={{ borderColor: 'var(--border-subtle)', my: 1 }} />
                  </Box>

                  <MenuItem onClick={handleLogout} sx={{ color: "var(--accent-error) !important" }}>
                    <ListItemIcon>
                      <Logout fontSize="small" sx={{ color: "var(--accent-error)" }} />
                    </ListItemIcon>
                    Logout
                  </MenuItem>
                </Menu>
              </>
            ) : (
              <div className="flex items-center gap-4">
                <Link
                   to="/login"
                   className="text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  to="/register"
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold text-[var(--color-obsidian-900)] transition-all duration-300 hover:shadow-lg hover:shadow-[var(--accent-primary)]/20 hover:-translate-y-0.5 active:translate-y-0"
                  style={{
                    background: 'linear-gradient(135deg, var(--color-insight-500) 0%, var(--color-aurora-500) 100%)',
                  }}
                >
                  Get Started
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
