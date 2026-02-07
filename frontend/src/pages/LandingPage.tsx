import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { HeroSection } from '../components/landing/HeroSection';
import { ValuePropositionGrid } from '../components/landing/ValuePropositionGrid';
import { SocialProof } from '../components/landing/SocialProof';
import { QuickStartCTA } from '../components/landing/QuickStartCTA';
import { Footer } from '../components/landing/Footer';

export default function LandingPage() {
  const { isAuthenticated, isInitialized } = useAuthStore();
  const navigate = useNavigate();

  useEffect(() => {
    if (isInitialized && isAuthenticated) {
      navigate('/dashboard', { replace: true });
    }
  }, [isInitialized, isAuthenticated, navigate]);

  if (isInitialized && isAuthenticated) {
    return null; // Or a loading spinner while redirecting
  }

  return (
    <div className="pt-16 bg-[var(--bg-page)] text-[var(--text-primary)]">
      <HeroSection />
      <ValuePropositionGrid />
      <SocialProof />
      <QuickStartCTA />
      <Footer />
    </div>
  );
}
