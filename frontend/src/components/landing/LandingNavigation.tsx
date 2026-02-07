import React from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import icon from '../../assets/sa-icon-192.png';

export const LandingNavigation: React.FC = () => {
  const { isAuthenticated } = useAuthStore();
  const navigate = useNavigate();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <header
      className="fixed top-0 w-full z-50 transition-all duration-300 bg-[#09090BD9] backdrop-blur-lg border-b border-[var(--border-subtle)]"
    >
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            to={isAuthenticated ? "/dashboard" : "/"}
            className="flex items-center no-underline gap-3 group"
            aria-label="Scholar Agent Home"
          >
            <div className="relative flex items-center justify-center">
              <div className="absolute inset-0 bg-[var(--accent-primary)] opacity-20 blur-md rounded-full group-hover:opacity-30 transition-opacity duration-300"></div>
              <img src={icon} alt="Scholar Agent" className="h-8 w-8 relative z-10" />
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
          <nav className="hidden md:flex items-center gap-8">
            <Link
              to="/how-it-works"
              className={`text-sm font-medium transition-colors duration-200 ${
                  isActive('/how-it-works') 
                    ? 'text-[var(--accent-primary)]' 
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
            >
              How It Works
            </Link>
            {isAuthenticated ? (
              <button
                onClick={() => navigate('/dashboard')}
                className="px-5 py-2.5 rounded-xl text-sm font-semibold text-[var(--color-obsidian-900)] transition-all duration-300 hover:shadow-lg hover:shadow-[var(--accent-primary)]/20 hover:-translate-y-0.5 active:translate-y-0"
                style={{
                  background: 'linear-gradient(135deg, var(--color-insight-500) 0%, var(--color-aurora-500) 100%)',
                }}
              >
                Dashboard
              </button>
            ) : (
              <div className="flex items-center gap-4">
                <button
                  onClick={() => navigate('/login')}
                  className="text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
                >
                  Sign In
                </button>
                <button
                  onClick={() => navigate('/register')}
                  className="px-5 py-2.5 rounded-xl text-sm font-semibold text-[var(--color-obsidian-900)] transition-all duration-300 hover:shadow-lg hover:shadow-[var(--accent-primary)]/20 hover:-translate-y-0.5 active:translate-y-0"
                  style={{
                    background: 'linear-gradient(135deg, var(--color-insight-500) 0%, var(--color-aurora-500) 100%)',
                  }}
                >
                  Get Started
                </button>
              </div>
            )}
          </nav>

          {/* Mobile Menu Button - Keeping simple for now, but aligned with theme */}
          <button
            className="md:hidden p-2 text-[var(--text-secondary)] hover:text-[var(--accent-primary)] transition-colors"
            aria-label="Menu"
            onClick={() => navigate('/register')}
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
};
