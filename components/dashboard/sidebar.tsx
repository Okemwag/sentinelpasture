"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Icon } from "@iconify/react";
import { useState } from "react";

const navigationItems = [
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
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <>
      {/* Mobile Menu Button */}
      <button
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 p-2 bg-white border border-[#E5E7EB] rounded-[8px] shadow-sm"
        aria-label="Toggle menu"
      >
        <Icon 
          icon={isMobileMenuOpen ? "mdi:close" : "mdi:menu"} 
          className="w-6 h-6 text-[#111111]" 
        />
      </button>

      {/* Overlay for mobile */}
      {isMobileMenuOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/20 z-30"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed left-0 top-0 h-screen w-64 bg-white border-r border-[#E5E7EB] z-40
          transition-transform duration-300 ease-in-out
          ${isMobileMenuOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}
        `}
      >
        <div className="p-6">
          <div className="flex items-center gap-2">
            <Icon icon="mdi:shield-check-outline" className="w-6 h-6 text-[#374151]" />
            <h1 className="text-[15px] font-medium text-[#111111]">
              National Risk Intelligence
            </h1>
          </div>
        </div>

        <nav className="px-3">
          {navigationItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className={`
                  flex items-center gap-3 px-3 py-2 text-[15px] font-normal rounded-[8px] mb-1
                  transition-colors duration-150
                  ${
                    isActive
                      ? "bg-[#F3F4F6] text-[#111111]"
                      : "text-[#6B7280] hover:bg-[#F9FAFB]"
                  }
                `}
              >
                <Icon icon={item.icon} className="w-5 h-5 flex-shrink-0" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* User Info at Bottom */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-[#E5E7EB]">
          <div className="flex items-center gap-3 px-2">
            <div className="w-8 h-8 rounded-full bg-[#F3F4F6] flex items-center justify-center">
              <Icon icon="mdi:account-outline" className="w-5 h-5 text-[#6B7280]" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[13px] font-medium text-[#111111] truncate">
                Administrator
              </div>
              <div className="text-[12px] text-[#6B7280]">
                Ministry Access
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
