"use client";

import { useEffect, useState } from "react";
import { Icon } from "@iconify/react";

import PageHeader from "@/components/dashboard/page-header";
import DateRangeSelector from "@/components/dashboard/date-range-selector";
import Panel from "@/components/dashboard/panel";
import StabilityIndex from "@/components/dashboard/stability-index";
import RiskMap from "@/components/dashboard/risk-map";
import PressureZonesTable from "@/components/dashboard/pressure-zones-table";
import { apiClient } from "@/lib/api-client";

type StabilityData = {
  value: number;
  trend: "up" | "down" | "stable";
  confidence: "High" | "Medium" | "Low";
  change: string;
};

type QuickStats = {
  activeAlerts: number;
  regionsMonitored: number;
  dataSources: number;
  lastUpdate: string;
};

type Driver = {
  label: string;
  percentage: number;
  trend: "up" | "down" | "stable";
  confidence: string;
};

export default function DashboardOverview() {
  const [stability, setStability] = useState<StabilityData | null>(null);
  const [stats, setStats] = useState<QuickStats | null>(null);
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [meta, setMeta] = useState({
    modelVersion: "loading",
    lastUpdated: "Loading...",
    confidence: "Medium",
  });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const [stabilityResponse, statsResponse, driversResponse] = await Promise.all([
          apiClient.getStabilityIndex(),
          apiClient.getQuickStats(),
          apiClient.getDrivers(),
        ]);

        if (!active) {
          return;
        }

        setStability(stabilityResponse.data);
        setStats(statsResponse.data);
        setDrivers(driversResponse.data);
        setMeta(
          stabilityResponse.metadata || {
            modelVersion: "unknown",
            lastUpdated: "unknown",
            confidence: "Medium",
          },
        );
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load dashboard");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader title="National Stability Overview">
        <DateRangeSelector />
      </PageHeader>

      {error ? (
        <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] p-4 text-[13px] text-[#991B1B]">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Active Alerts"
          value={stats ? String(stats.activeAlerts) : "..."}
          note="Requires attention"
          icon="mdi:alert-circle-outline"
        />
        <StatCard
          label="Regions Monitored"
          value={stats ? String(stats.regionsMonitored) : "..."}
          note="Current feature snapshots"
          icon="mdi:map-marker-multiple-outline"
        />
        <StatCard
          label="Data Sources"
          value={stats ? String(stats.dataSources) : "..."}
          note="Model feature inputs"
          icon="mdi:database-outline"
        />
        <StatCard
          label="Last Update"
          value={stats ? formatTimestamp(stats.lastUpdate) : "..."}
          note="From latest model snapshot"
          icon="mdi:clock-outline"
        />
      </div>

      <Panel
        title="National Risk Summary"
        metadata={{
          modelVersion: meta.modelVersion,
          lastUpdated: meta.lastUpdated,
          confidence: meta.confidence,
          showExplainLink: true,
        }}
      >
        <StabilityIndex
          value={stability?.value ?? 0}
          trend={stability?.trend ?? "stable"}
          confidence={stability?.confidence ?? "Medium"}
          change={stability?.change ?? "Loading"}
        />
      </Panel>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel title="Regional Risk Distribution">
          <RiskMap />
        </Panel>

        <Panel title="Risk Factors Summary">
          <div className="space-y-4">
            {(drivers.length ? drivers : placeholderDrivers).map((driver) => (
              <div key={driver.label} className="flex items-start gap-3">
                <Icon icon={iconForTrend(driver.trend)} className="mt-0.5 h-5 w-5 flex-shrink-0 text-[#6B7280]" />
                <div>
                  <div className="mb-1 text-[15px] font-medium text-[#111111]">
                    {driver.label}
                  </div>
                  <div className="text-[13px] leading-relaxed text-[#6B7280]">
                    Contribution {driver.percentage.toFixed(1)}%. Confidence {driver.confidence}.
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel title="Top Emerging Pressure Zones">
        <PressureZonesTable />
      </Panel>
    </div>
  );
}

function StatCard({
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

function iconForTrend(trend: Driver["trend"]) {
  if (trend === "up") {
    return "mdi:trending-up";
  }
  if (trend === "down") {
    return "mdi:trending-down";
  }
  return "mdi:trending-neutral";
}

function formatTimestamp(value: string) {
  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) {
    return value;
  }
  return timestamp.toLocaleDateString();
}

const placeholderDrivers: Driver[] = [
  {
    label: "Loading drivers",
    percentage: 0,
    trend: "stable",
    confidence: "Pending",
  },
];
