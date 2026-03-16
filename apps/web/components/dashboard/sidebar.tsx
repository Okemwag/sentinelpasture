"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Icon } from "@iconify/react";
import { useEffect, useState } from "react";
import { clearAuthSession, readAuthSession } from "@/lib/auth-session";

const NAVIGATION = [
  { label: "Overview", href: "/dashboard", icon: "mdi:view-dashboard-outline" },
  { label: "Alerts", href: "/dashboard/alerts", icon: "mdi:bell-outline" },
  { label: "Regional Risk", href: "/dashboard/regional", icon: "mdi:map-marker-radius-outline" },
  { label: "Drivers", href: "/dashboard/drivers", icon: "mdi:chart-timeline-variant" },
  { label: "Interventions", href: "/dashboard/interventions", icon: "mdi:strategy" },
  { label: "Outcomes", href: "/dashboard/outcomes", icon: "mdi:chart-line" },
  { label: "Reports", href: "/dashboard/reports", icon: "mdi:file-document-outline" },
  { label: "Settings", href: "/dashboard/settings", icon: "mdi:cog-outline" },
];

export default function DashboardSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [identity, setIdentity] = useState({ full_name: "Authorized User", role: "Restricted Access" });
  const [time, setTime] = useState<string>("");

  useEffect(() => {
    const session = readAuthSession();
    if (session) {
      setIdentity({
        full_name: session.user.full_name,
        role: `${session.user.role.charAt(0).toUpperCase()}${session.user.role.slice(1)} Access`,
      });
    }

    const tick = () =>
      setTime(
        new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
      );
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, []);

  function handleSignOut() {
    clearAuthSession();
    router.push("/signin");
  }

  return (
    <>
      {/* Mobile toggle */}
      <button
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white border border-[#E5E7EB] rounded-[8px] shadow-sm"
        aria-label="Toggle menu"
      >
        <Icon icon={isMobileMenuOpen ? "mdi:close" : "mdi:menu"} className="w-6 h-6 text-[#111111]" />
      </button>

      {isMobileMenuOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/20 z-30"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      <aside
        className={`
          fixed left-0 top-0 h-screen w-64 bg-white border-r border-[#E5E7EB] z-40
          flex flex-col
          transition-transform duration-300 ease-in-out
          ${isMobileMenuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
        `}
      >
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#F3F4F6]">
          <div className="flex items-center gap-2.5">
            <div className="relative flex items-center justify-center w-8 h-8 rounded-[6px] bg-[#F3F4F6]">
              <Icon icon="mdi:shield-check-outline" className="w-5 h-5 text-[#374151]" />
              {/* Live pulse dot */}
              <span className="absolute -top-0.5 -right-0.5 flex h-2 w-2">
                <span className="animate-live absolute inline-flex h-full w-full rounded-full bg-[#3A6B33] opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-[#3A6B33]" />
              </span>
            </div>
            <div>
              <div className="text-[13px] font-semibold text-[#111111] leading-tight">
                National Risk Intel
              </div>
              <div className="text-[10px] text-[#9CA3AF] tracking-widest uppercase">
                Resilience Platform
              </div>
            </div>
          </div>
          {/* Live clock */}
          {time && (
            <div className="mt-2.5 text-[11px] font-mono text-[#9CA3AF] tracking-wider px-1">
              {time}
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-3 overflow-y-auto">
          <div className="text-[10px] uppercase tracking-[0.18em] text-[#9CA3AF] px-3 mb-2">
            Intelligence
          </div>
          {NAVIGATION.slice(0, 5).map((item) => (
            <NavItem
              key={item.href}
              {...item}
              isActive={pathname === item.href}
              onClick={() => setIsMobileMenuOpen(false)}
            />
          ))}
          <div className="text-[10px] uppercase tracking-[0.18em] text-[#9CA3AF] px-3 mt-4 mb-2">
            Management
          </div>
          {NAVIGATION.slice(5).map((item) => (
            <NavItem
              key={item.href}
              {...item}
              isActive={pathname === item.href}
              onClick={() => setIsMobileMenuOpen(false)}
            />
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-[#F3F4F6] px-4 py-3">
          <div className="flex items-center gap-2.5 px-1 mb-2.5">
            <div className="w-7 h-7 rounded-full bg-[#F3F4F6] flex items-center justify-center shrink-0">
              <Icon icon="mdi:account-outline" className="w-4 h-4 text-[#6B7280]" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-[12px] font-medium text-[#111111] truncate">
                {identity.full_name}
              </div>
              <div className="text-[10px] text-[#9CA3AF]">{identity.role}</div>
            </div>
          </div>
          <button
            type="button"
            onClick={handleSignOut}
            className="flex items-center justify-center gap-2 w-full px-3 py-2 text-[12px] text-[#6B7280] border border-[#E5E7EB] rounded-[8px] hover:bg-[#F9FAFB] transition-colors"
          >
            <Icon icon="mdi:logout" className="w-4 h-4" />
            <span>Sign Out</span>
          </button>
        </div>
      </aside>
    </>
  );
}

function NavItem({
  label,
  href,
  icon,
  isActive,
  onClick,
}: {
  label: string;
  href: string;
  icon: string;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <Link
      href={href}
      onClick={onClick}
      className={`
        flex items-center gap-2.5 px-3 py-2 text-[13px] rounded-[8px] mb-0.5
        transition-colors duration-150 relative
        ${isActive
          ? "bg-[#F3F4F6] text-[#111111] font-medium"
          : "text-[#6B7280] hover:bg-[#F9FAFB] hover:text-[#374151]"
        }
      `}
    >
      {isActive && (
        <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 bg-[#374151] rounded-r-full" />
      )}
      <Icon icon={icon} className="w-4 h-4 shrink-0" />
      <span>{label}</span>
    </Link>
  );
}
