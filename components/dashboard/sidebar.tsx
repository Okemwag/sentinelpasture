"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigationItems = [
  { label: "Overview", href: "/dashboard" },
  { label: "Regional Risk", href: "/dashboard/regional" },
  { label: "Drivers", href: "/dashboard/drivers" },
  { label: "Interventions", href: "/dashboard/interventions" },
  { label: "Outcomes", href: "/dashboard/outcomes" },
  { label: "Reports", href: "/dashboard/reports" },
  { label: "Settings", href: "/dashboard/settings" },
];

export default function DashboardSidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-white border-r border-[#E5E7EB]">
      <div className="p-6">
        <h1 className="text-[15px] font-medium text-[#111111]">
          National Risk Intelligence
        </h1>
      </div>
      
      <nav className="px-3">
        {navigationItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                block px-3 py-2 text-[15px] font-normal rounded-[8px] mb-1
                transition-colors duration-150
                ${
                  isActive
                    ? "bg-[#F3F4F6] text-[#111111]"
                    : "text-[#6B7280] hover:bg-[#F9FAFB]"
                }
              `}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
