import { Inter } from "next/font/google";
import DashboardSidebar from "@/components/dashboard/sidebar";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
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
      <div className="flex min-h-screen bg-[#FAFAFA]">
        <DashboardSidebar />
        <main className="flex-1 lg:ml-64 w-full">
          <div className="max-w-[1400px] mx-auto p-4 sm:p-6 lg:p-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
