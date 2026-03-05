"use client";

import { useEffect, useState } from "react";
import { Icon } from "@iconify/react";

import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import { apiClient } from "@/lib/api-client";

type AlertRow = {
  id: number;
  severity: "elevated" | "moderate";
  title: string;
  description: string;
  timestamp: string;
  region: string;
  status: "active" | "monitoring" | "resolved";
};

type AlertStats = {
  active: number;
  monitoring: number;
  resolved24h: number;
};

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [stats, setStats] = useState<AlertStats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const [alertsResponse, statsResponse] = await Promise.all([
          apiClient.getAlerts(),
          apiClient.getAlertStats(),
        ]);
        if (!active) {
          return;
        }
        setAlerts(alertsResponse.data);
        setStats(statsResponse.data);
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load alerts");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader title="National Alerts & Notifications" />

      {error ? (
        <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] p-4 text-[13px] text-[#991B1B]">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <SummaryCard label="Active Alerts" value={String(stats?.active ?? 0)} note="Requires attention" icon="mdi:alert-circle-outline" />
        <SummaryCard label="Monitoring" value={String(stats?.monitoring ?? 0)} note="Under observation" icon="mdi:eye-outline" />
        <SummaryCard label="Resolved (24h)" value={String(stats?.resolved24h ?? 0)} note="Last day" icon="mdi:check-circle-outline" />
      </div>

      <Panel title="Recent Alerts">
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className="rounded-[8px] border border-[#E5E7EB] p-4 transition-colors duration-150 hover:bg-[#F9FAFB]"
            >
              <div className="flex items-start gap-3">
                <div className="mt-1 flex-shrink-0">
                  <Icon
                    icon={alert.severity === "elevated" ? "mdi:alert-circle-outline" : "mdi:information-outline"}
                    className="h-6 w-6 text-[#6B7280]"
                  />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="mb-2 flex items-start justify-between gap-4">
                    <h3 className="text-[15px] font-medium text-[#111111]">{alert.title}</h3>
                    <span className="flex-shrink-0 rounded-[4px] bg-[#F9FAFB] px-2 py-1 text-[12px] text-[#6B7280]">
                      {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                    </span>
                  </div>

                  <p className="mb-3 text-[13px] leading-relaxed text-[#6B7280]">{alert.description}</p>

                  <div className="flex flex-wrap items-center gap-4 text-[12px] text-[#6B7280]">
                    <div className="flex items-center gap-1">
                      <Icon icon="mdi:map-marker-outline" className="h-4 w-4" />
                      <span>{alert.region}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Icon icon="mdi:clock-outline" className="h-4 w-4" />
                      <span>{new Date(alert.timestamp).toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  note,
  icon,
}: {
  label: string;
  value: string;
  note: string;
  icon: string;
}) {
  return (
    <div className="rounded-[8px] border border-[#E5E7EB] bg-white p-4">
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[13px] text-[#6B7280]">{label}</span>
        <Icon icon={icon} className="h-5 w-5 text-[#6B7280]" />
      </div>
      <div className="text-[24px] font-semibold text-[#111111]">{value}</div>
      <div className="mt-1 text-[12px] text-[#6B7280]">{note}</div>
    </div>
  );
}
