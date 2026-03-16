import { Inter } from "next/font/google";
import AuthGate from "@/components/auth-gate";
import DashboardSidebar from "@/components/dashboard/sidebar";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-inter",
});

export const metadata = {
  title: "National Risk Intelligence & Resilience Platform",
  description: "Governance-grade risk intelligence dashboard",
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className={`${inter.variable} font-sans min-h-screen bg-[#F4F4F2]`}>
      <AuthGate>
        <div className="flex min-h-screen">
          <DashboardSidebar />
          {/* Sidebar is fixed-width 256px via the sidebar component; push content right */}
          <main className="flex-1 min-w-0" style={{ marginLeft: 256 }}>
            {children}
          </main>
        </div>
      </AuthGate>
    </div>
  );
}
