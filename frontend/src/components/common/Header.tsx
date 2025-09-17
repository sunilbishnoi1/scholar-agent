import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { AccountCircle, Logout, Build as BuildIcon, Info as InfoIcon } from '@mui/icons-material';
import { IconButton, Button, Box, Menu, MenuItem, ListItemIcon, Typography, Divider } from '@mui/material';
import { useAuthStore } from '../../store/authStore';
import icon from '../../assets/sa-icon-192.png'

const Header = () => {
    const { isAuthenticated, logout, user, fetchUser } = useAuthStore();
    const navigate = useNavigate();
    const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
    const open = Boolean(anchorEl);

    // This effect runs when the app loads. If the user is marked as authenticated
    // but their profile data hasn't been loaded yet, it fetches it.
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
        navigate('/login');
    };

    const handleNavigate = (path: string) => {
        handleClose();
        navigate(path);
    };

    return (
        <header className="fixed top-0 w-full  backdrop-blur-sm  bg-opacity-90 border-b border-gray-800 z-50">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <Link to="/" className="flex items-center no-underline text-slate-800">
                        <img src={icon} className="h-8 w-8" />
                        <span className="bg-gradient-to-r from-blue-600 to-teal-500 bg-clip-text text-transparent ml-3 font-bold text-xl tracking-normal font-poppins">
                            Scholar Agent
                        </span>
                    </Link>

                    {/* START: ADDED DESKTOP NAVIGATION */}
                    {isAuthenticated && (
                        <Box sx={{
                            display: { xs: 'none', md: 'flex' },
                            position: 'absolute',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            gap: 2 
                        }}>
                            <Button component={Link} to="/tools" sx={{ color: 'text.primary', fontWeight: 500 }}>
                                Tools
                            </Button>
                            <Button component={Link} to="/know" sx={{ color: 'text.primary', fontWeight: 500 }}>
                                Know
                            </Button>
                        </Box>
                    )}
                    {!isAuthenticated && (
                    <Box sx={{
                            display: { xs: 'none', md: 'flex' },
                            position: 'absolute',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            gap: 2 
                        }}>
                            
                            <Button
                                component={Link}
                                to="/know"
                                sx={{
                                    color: 'text.primary',
                                    fontWeight: 700, // bolder for emphasis
                                    fontSize: { xs: '0.95rem', md: '1.05rem', lg: '1.15rem' }, // responsive sizing
                                    letterSpacing: '0.5px', // subtle spacing for clarity
                                    textTransform: 'none', // keeps original casing
                                }}
                                >
                                Features
                                </Button>
                        </Box>
                    )}
                    {/* END: ADDED DESKTOP NAVIGATION */}

                    <div>
                        {isAuthenticated ? (
                            <>
                                <IconButton
                                    size="large"
                                    aria-label="account of current user"
                                    aria-controls="profile-menu"
                                    aria-haspopup="true"
                                    onClick={handleMenu}
                                    color="inherit"
                                >
                                    <AccountCircle sx={{ fontSize: '1.75rem', color: '#2563eb' }} />
                                </IconButton>
                                <Menu
                                    id="profile-menu"
                                    anchorEl={anchorEl}
                                    anchorOrigin={{
                                        vertical: 'bottom',
                                        horizontal: 'right',
                                    }}
                                    keepMounted
                                    transformOrigin={{
                                        vertical: 'top',
                                        horizontal: 'right',
                                    }}
                                    open={open}
                                    onClose={handleClose}
                                    PaperProps={{
                                      elevation: 0,
                                      sx: {
                                        overflow: 'visible',
                                        filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.32))',
                                        mt: 1.5,
                                        minWidth: 220,
                                        backgroundColor: 'rgba(255, 255, 255, 0.6)', 
                                        backdropFilter: 'blur(12px)', 
                                        WebkitBackdropFilter: 'blur(12px)',
                                        '&:before': {
                                          content: '""',
                                          display: 'block',
                                          position: 'absolute',
                                          top: 0,
                                          right: 14,
                                          width: 10,
                                          height: 10,
                                          bgcolor: 'background.paper',
                                          transform: 'translateY(-50%) rotate(45deg)',
                                          zIndex: 0,
                                        },
                                      },
                                    }}
                                >
                                    {/* Mobile-only "Profile" title */}
                                    <Typography sx={{ display: { xs: 'block', md: 'none' }, px: 2, pt: 1, pb: 0, fontWeight: 500, color: '#2563eb' }}>
                                        Profile
                                    </Typography>
                                    
                                    <Box sx={{ px: 2, py: 1.5 }}>
                                        <Typography variant="body1" fontWeight="bold">{user?.name || 'Loading...'}</Typography>
                                        <Typography variant="body2" color="text.secondary">{user?.email || '...'}</Typography>
                                    </Box>
                                    
                                    {/* START: ADDED MOBILE NAVIGATION */}
                                    <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                                        <Divider sx={{ my: 0.5 }} />
                                        <MenuItem onClick={() => handleNavigate('/tools')}>
                                            <ListItemIcon>
                                                <BuildIcon fontSize="small" />
                                            </ListItemIcon>
                                            Tools
                                        </MenuItem>
                                        <MenuItem onClick={() => handleNavigate('/know')}>
                                            <ListItemIcon>
                                                <InfoIcon fontSize="small" />
                                            </ListItemIcon>
                                            Know
                                        </MenuItem>
                                    </Box>
                                    {/* END: ADDED MOBILE NAVIGATION */}

                                    <Divider sx={{ my: 0.5 }} />
                                    
                                    <MenuItem onClick={handleLogout} sx={{color:'red'}}>
                                        <ListItemIcon>
                                            <Logout fontSize="small" sx={{color:'red'}} />
                                        </ListItemIcon>
                                        Logout
                                    </MenuItem>
                                </Menu>
                            </>
                        ) : (
                            <Box>
                                <Button variant="contained" component={Link} className='bg-gradient-to-r from-blue-600 to-teal-500 hover:bg-blue-700 text-white' to="/register">Get Started</Button>
                            </Box>
                        )}
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;