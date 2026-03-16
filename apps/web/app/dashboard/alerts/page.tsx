"use client";

import { useEffect, useState } from "react";
import { Icon } from "@iconify/react";
import PageHeader from "@/components/dashboard/page-header";
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

const SEVERITY_CONFIG = {
  elevated: {
    border: "#8C6A3D",
    bg: "#F5EFE6",
    dot: "#8C6A3D",
    label: "High",
    icon: "mdi:alert-circle-outline",
  },
  moderate: {
    border: "#C7A56B",
    bg: "#FBF5EB",
    dot: "#C7A56B",
    label: "Elevated",
    icon: "mdi:information-outline",
  },
};

const STATUS_CONFIG = {
  active: { bg: "#EDE8E0", text: "#2A1E12", label: "Active" },
  monitoring: { bg: "#FBF5EB", text: "#7A4F1E", label: "Monitoring" },
  resolved: { bg: "#EDF4EB", text: "#3A6B33", label: "Resolved" },
};

function urgencyScore(alert: AlertRow): number {
  const age = Date.now() - new Date(alert.timestamp).getTime();
  const agePenalty = Math.min(age / (1000 * 60 * 60 * 48), 1); // decay over 48h
  const baseSeverity = alert.severity === "elevated" ? 85 : 60;
  const statusBonus = alert.status === "active" ? 0 : alert.status === "monitoring" ? -15 : -40;
  return Math.max(0, Math.round(baseSeverity + statusBonus - agePenalty * 20));
}

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [stats, setStats] = useState<AlertStats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    Promise.all([apiClient.getAlerts(), apiClient.getAlertStats()])
      .then(([a, s]) => {
        if (!active) return;
        setAlerts(a.data);
        setStats(s.data);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load alerts");
      });
    return () => { active = false; };
  }, []);

  // Build timeline density (last 7 days)
  const timeline = buildTimeline(alerts);

  return (
    <div className="max-w-[1400px] mx-auto p-6 space-y-6">
      <PageHeader title="National Alerts & Notifications" />

      {error && (
        <div className="rounded-[8px] border border-[#E2C99A] bg-[#FBF5EB] p-3 text-[13px] text-[#7A4F1E]">
          {error}
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard icon="mdi:alert-circle-outline" label="Active" value={String(stats?.active ?? 0)}
          note="Requires attention" color="#8C6A3D" bg="#F5EFE6" />
        <StatCard icon="mdi:eye-outline" label="Monitoring" value={String(stats?.monitoring ?? 0)}
          note="Under observation" color="#C7A56B" bg="#FBF5EB" />
        <StatCard icon="mdi:check-circle-outline" label="Resolved (24h)" value={String(stats?.resolved24h ?? 0)}
          note="Last 24 hours" color="#3A6B33" bg="#EDF4EB" />
      </div>

      {/* Alert density timeline */}
      {timeline.length > 0 && (
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-4">
          <div className="text-[11px] uppercase tracking-[0.18em] text-[#9CA3AF] mb-3">
            Alert Activity — Last 7 Days
          </div>
          <div className="flex items-end gap-1 h-12">
            {timeline.map((day) => (
              <div key={day.label} className="flex-1 flex flex-col items-center gap-1">
                <div
                  className="w-full rounded-t-[2px] transition-all"
                  style={{
                    height: `${(day.count / Math.max(...timeline.map((d) => d.count), 1)) * 40}px`,
                    backgroundColor: day.count > 2 ? "#8C6A3D" : day.count > 0 ? "#C7A56B" : "#F3F4F6",
                    minHeight: day.count > 0 ? "4px" : "1px",
                  }}
                />
                <span className="text-[9px] text-[#9CA3AF]">{day.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Alert list */}
      <div className="bg-white border border-[#E5E7EB] rounded-[8px]">
        <div className="border-b border-[#F3F4F6] px-5 py-3">
          <h2 className="text-[15px] font-medium text-[#111111]">Recent Alerts</h2>
        </div>
        <div className="divide-y divide-[#F3F4F6]">
          {(alerts.length ? alerts : PLACEHOLDER_ALERTS).map((alert) => {
            const sev = SEVERITY_CONFIG[alert.severity];
            const st = STATUS_CONFIG[alert.status];
            const score = urgencyScore(alert);
            return (
              <div
                key={alert.id}
                className="flex items-start gap-0 hover:bg-[#FAFAFA] transition-colors"
                style={{ borderLeft: `3px solid ${sev.border}` }}
              >
                {/* Severity dot column */}
                <div className="flex flex-col items-center px-4 pt-4 pb-4 shrink-0">
                  <Icon icon={sev.icon} className="h-5 w-5" style={{ color: sev.dot }} />
                  <div className="mt-1.5 text-[10px] font-semibold uppercase tracking-widest" style={{ color: sev.dot }}>
                    {sev.label}
                  </div>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 py-4 pr-4">
                  <div className="flex items-start justify-between gap-3 mb-1">
                    <h3 className="text-[14px] font-semibold text-[#111111]">{alert.title}</h3>
                    <div className="flex items-center gap-2 shrink-0">
                      {/* Urgency score */}
                      <div className="text-right">
                        <div className="text-[10px] text-[#9CA3AF] uppercase tracking-widest">Urgency</div>
                        <div className="text-[14px] font-bold" style={{ color: score >= 70 ? "#4A3A26" : score >= 50 ? "#8C6A3D" : "#C7A56B" }}>
                          {score}
                        </div>
                      </div>
                      <span
                        className="text-[11px] px-2.5 py-1 rounded-full font-medium"
                        style={{ backgroundColor: st.bg, color: st.text }}
                      >
                        {st.label}
                      </span>
                    </div>
                  </div>
                  <p className="text-[13px] text-[#6B7280] leading-relaxed mb-2">{alert.description}</p>
                  <div className="flex flex-wrap gap-4 text-[12px] text-[#9CA3AF]">
                    <span className="flex items-center gap-1">
                      <Icon icon="mdi:map-marker-outline" className="h-3.5 w-3.5" />
                      {alert.region}
                    </span>
                    <span className="flex items-center gap-1">
                      <Icon icon="mdi:clock-outline" className="h-3.5 w-3.5" />
                      {new Date(alert.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, note, color, bg }: {
  icon: string; label: string; value: string; note: string; color: string; bg: string;
}) {
  return (
    <div className="rounded-[8px] border border-[#E5E7EB] bg-white p-4 flex items-center gap-4">
      <div className="h-10 w-10 rounded-full flex items-center justify-center shrink-0"
        style={{ backgroundColor: bg }}>
        <Icon icon={icon} className="h-5 w-5" style={{ color }} />
      </div>
      <div>
        <div className="text-[12px] text-[#9CA3AF]">{label}</div>
        <div className="text-[24px] font-bold text-[#111111] leading-tight">{value}</div>
        <div className="text-[11px] text-[#9CA3AF]">{note}</div>
      </div>
    </div>
  );
}

function buildTimeline(alerts: AlertRow[]) {
  const days: { label: string; count: number }[] = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const label = d.toLocaleDateString("en-US", { weekday: "short" }).slice(0, 2);
    const dateStr = d.toDateString();
    const count = alerts.filter((a) => new Date(a.timestamp).toDateString() === dateStr).length;
    days.push({ label, count });
  }
  return days;
}

const PLACEHOLDER_ALERTS: AlertRow[] = [
  {
    id: 1,
    severity: "elevated",
    title: "Flood Risk Escalation — North Eastern Drylands",
    description: "Satellite rainfall anomaly at +210% above seasonal norm. River overflow indicators active.",
    timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    region: "North Eastern Drylands",
    status: "active",
  },
  {
    id: 2,
    severity: "moderate",
    title: "Livestock Movement Spike — Coast Belt",
    description: "Abnormal pastoral migration patterns detected. Market disruption risk elevating.",
    timestamp: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(),
    region: "Coast Belt",
    status: "monitoring",
  },
  {
    id: 3,
    severity: "moderate",
    title: "Water Point Pressure — South Rift Valley",
    description: "Local mediation requests increasing. Water access stress above seasonal baseline.",
    timestamp: new Date(Date.now() - 18 * 60 * 60 * 1000).toISOString(),
    region: "South Rift Valley",
    status: "monitoring",
  },
];
