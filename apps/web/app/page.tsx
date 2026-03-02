import HeroSection from "@/components/hero-section";
import Features from "@/components/features-1";
import PlatformVisual from "@/components/platform-visual";
import ContentSection from "@/components/content-1";
import GovernanceGuardrails from "@/components/governance-guardrails";
import CaseStudies from "@/components/case-studies";
import CallToAction from "@/components/call-to-action";
import FooterSection from "@/components/footer";

export default function Home() {
  return (
    <>
      <HeroSection />
      <Features />
      <PlatformVisual />
      <ContentSection />
      <GovernanceGuardrails />
      <CaseStudies />
      <CallToAction />
      <FooterSection />
    </>
  );
}
