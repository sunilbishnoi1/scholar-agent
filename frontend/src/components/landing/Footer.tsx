import React from 'react';
import { Link } from 'react-router-dom';
import {
  GitHub as GitHubIcon,
  Twitter as TwitterIcon,
  LinkedIn as LinkedInIcon,
  Email as EmailIcon,
} from '@mui/icons-material';

export const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { label: 'Features', path: '/how-it-works' },
      { label: 'Pricing', path: '/pricing' },
      { label: 'Documentation', path: '/docs' },
    ],
    company: [
      { label: 'About', path: '/about' },
      { label: 'Blog', path: '/blog' },
      { label: 'Careers', path: '/careers' },
    ],
    legal: [
      { label: 'Privacy Policy', path: '/privacy' },
      { label: 'Terms of Service', path: '/terms' },
      { label: 'Cookie Policy', path: '/cookies' },
    ],
    support: [
      { label: 'Contact', path: '/contact' },
      { label: 'Help Center', path: '/help' },
      { label: 'Community', path: '/community' },
    ],
  };

  const socialLinks = [
    { icon: <GitHubIcon />, href: 'https://github.com', label: 'GitHub' },
    { icon: <TwitterIcon />, href: 'https://twitter.com', label: 'Twitter' },
    { icon: <LinkedInIcon />, href: 'https://linkedin.com', label: 'LinkedIn' },
    { icon: <EmailIcon />, href: 'mailto:support@scholaragent.com', label: 'Email' },
  ];

  return (
    <footer className="bg-[var(--bg-page)] text-[var(--text-primary)] py-16 px-4 sm:px-6 lg:px-8">
      <div className="container mx-auto max-w-7xl">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-12 mb-12">
          {/* Brand Section */}
          <div className="lg:col-span-2">
            <Link to="/" className="flex items-center mb-4 no-underline">
              <span
                className="text-2xl font-bold"
                style={{
                  background:
                    'linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                  fontFamily: 'var(--font-primary)',
                }}
              >
                Scholar Agent
              </span>
            </Link>
            <p
              className="text-[var(--text-secondary)] mb-6 max-w-md"
              style={{ fontFamily: 'var(--font-content)' }}
            >
              Transforming academic research with AI-powered multi-agent systems. Accelerate your discoveries and unlock new insights.
            </p>
            {/* Social Links */}
            <div className="flex gap-4">
              {socialLinks.map((social, index) => (
                <a
                  key={index}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 flex items-center justify-center rounded-[var(--radius-md)] bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:bg-[var(--accent-secondary)] hover:text-[var(--bg-page)] transition-all duration-[var(--transition-standard)]"
                  aria-label={social.label}
                >
                  {social.icon}
                </a>
              ))}
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3
              className="font-semibold mb-4 text-[var(--text-primary)]"
              style={{ fontFamily: 'var(--font-primary)' }}
            >
              Product
            </h3>
            <ul className="space-y-3">
              {footerLinks.product.map((link, index) => (
                <li key={index}>
                  <Link
                    to={link.path}
                    className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors duration-[var(--transition-fast)] no-underline"
                    style={{ fontFamily: 'var(--font-primary)' }}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h3
              className="font-semibold mb-4 text-[var(--text-primary)]"
              style={{ fontFamily: 'var(--font-primary)' }}
            >
              Company
            </h3>
            <ul className="space-y-3">
              {footerLinks.company.map((link, index) => (
                <li key={index}>
                  <Link
                    to={link.path}
                    className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors duration-[var(--transition-fast)] no-underline"
                    style={{ fontFamily: 'var(--font-primary)' }}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Legal & Support */}
          <div>
            <h3
              className="font-semibold mb-4 text-[var(--text-primary)]"
              style={{ fontFamily: 'var(--font-primary)' }}
            >
              Legal
            </h3>
            <ul className="space-y-3 mb-6">
              {footerLinks.legal.map((link, index) => (
                <li key={index}>
                  <Link
                    to={link.path}
                    className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors duration-[var(--transition-fast)] no-underline"
                    style={{ fontFamily: 'var(--font-primary)' }}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
            <h3
              className="font-semibold mb-4 text-white mt-6"
              style={{ fontFamily: 'var(--font-primary)' }}
            >
              Support
            </h3>
            <ul className="space-y-3">
              {footerLinks.support.map((link, index) => (
                <li key={index}>
                  <Link
                    to={link.path}
                    className="text-[var(--color-slate-400)] hover:text-white transition-colors duration-[var(--transition-fast)] no-underline"
                    style={{ fontFamily: 'var(--font-primary)' }}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-[var(--border-subtle)] pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p
            className="text-[var(--text-secondary)] text-sm"
            style={{ fontFamily: 'var(--font-primary)' }}
          >
            © {currentYear} Scholar Agent. All rights reserved.
          </p>
          <p
            className="text-[var(--text-secondary)] text-sm"
            style={{ fontFamily: 'var(--font-primary)' }}
          >
            Built with ❤️ for researchers worldwide
          </p>
        </div>
      </div>
    </footer>
  );
};
