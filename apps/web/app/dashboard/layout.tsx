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
    <div className={`${inter.variable} font-sans`}>
      <AuthGate>
        <div className="flex min-h-screen bg-[#F7F7F5]">
          <DashboardSidebar />
          <main className="flex-1 lg:ml-64 w-full min-w-0 overflow-hidden">
            {children}
          </main>
        </div>
      </AuthGate>
    </div>
  );
}
