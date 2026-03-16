"use client";

import { useEffect, useState, useCallback } from "react";
import { Icon } from "@iconify/react";

import RiskMap, { type MapRegion } from "@/components/dashboard/risk-map";
import IntelligencePanel from "@/components/dashboard/intelligence-panel";
import AnalyticsStrip from "@/components/dashboard/analytics-strip";
import { apiClient } from "@/lib/api-client";

type QuickStats = {
  activeAlerts: number;
  regionsMonitored: number;
  dataSources: number;
  lastUpdate: string;
};

type StabilityData = {
  value: number;
  trend: "up" | "down" | "stable";
  confidence: "High" | "Medium" | "Low";
  change: string;
};

type ChartPoint = { date: string; value: number };
type RegionComparison = { region: string; riskScore: number; riskLevel: string };

function formatTime(value: string) {
  const d = new Date(value);
  if (isNaN(d.getTime())) return value;
  return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
}

function formatDate(value: string) {
  const d = new Date(value);
  if (isNaN(d.getTime())) return value;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

const TREND_ICONS = {
  up: "mdi:trending-up",
  down: "mdi:trending-down",
  stable: "mdi:trending-neutral",
};

function TrendLabel({ trend, value }: { trend: "up" | "down" | "stable"; value: number }) {
  const color = trend === "up" ? "#8C6A3D" : trend === "down" ? "#3A6B33" : "#6B7280";
  return (
    <span className="flex items-center gap-1 text-[11px]" style={{ color }}>
      <Icon icon={TREND_ICONS[trend]} className="h-3.5 w-3.5" />
      {Math.round(value)}
    </span>
  );
}

export default function DashboardOverview() {
  const [stats, setStats] = useState<QuickStats | null>(null);
  const [stability, setStability] = useState<StabilityData | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<MapRegion | null>(null);
  const [trendData, setTrendData] = useState<ChartPoint[]>([]);
  const [comparisonData, setComparisonData] = useState<RegionComparison[]>([]);
  const [meta, setMeta] = useState({ lastUpdated: "Loading…", modelVersion: "—" });

  useEffect(() => {
    Promise.all([apiClient.getQuickStats(), apiClient.getStabilityIndex()]).then(
      ([s, stab]) => {
        setStats(s.data);
        setStability(stab.data);
        setMeta(stab.metadata ?? { lastUpdated: "—", modelVersion: "—", confidence: "Medium" });
      }
    ).catch(() => {
      // Use fallback values silently
    });

    apiClient.getOutcomesChart().then((r) => setTrendData(r.data)).catch(() => { });

    apiClient.getRegionalData().then((r) => {
      const top = r.data
        .slice()
        .sort((a, b) => b.riskScore - a.riskScore)
        .slice(0, 8)
        .map((d) => ({
          region: d.region,
          riskScore: d.riskScore,
          riskLevel: d.riskLevel,
        }));
      setComparisonData(top);
    }).catch(() => { });
  }, []);

  const handleRegionSelect = useCallback((region: MapRegion) => {
    setSelectedRegion(region);
  }, []);

  const stabilityScore = stability?.value ?? 0;
  const stabilityColor =
    stabilityScore >= 70 ? "#4A3A26" : stabilityScore >= 50 ? "#8C6A3D" : stabilityScore >= 30 ? "#C7A56B" : "#7A9B70";

  return (
    <div className="flex flex-col gap-0 bg-[#F7F7F5] min-h-screen">

      {/* ═══════════════ TOP BAR ═══════════════ */}
      <div className="border-b border-[#E5E7EB] bg-white px-6 py-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-[16px] font-semibold text-[#111111]">National Stability Overview</h1>
            <div className="flex items-center gap-3 mt-0.5">
              <span className="text-[12px] text-[#9CA3AF]">Model: {meta.modelVersion}</span>
              <span className="text-[12px] text-[#9CA3AF]">Updated: {formatDate(meta.lastUpdated)}</span>
              <span className="flex items-center gap-1.5 text-[12px] text-[#3A6B33]">
                <span className="relative flex h-2 w-2">
                  <span className="animate-live absolute inline-flex h-full w-full rounded-full bg-[#3A6B33] opacity-75" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-[#3A6B33]" />
                </span>
                Live
              </span>
            </div>
          </div>

          {/* Stat cards inline */}
          <div className="flex flex-wrap gap-3">
            {[
              {
                label: "Stability Index",
                value: stability ? String(Math.round(stability.value)) : "—",
                sub: stability ? <TrendLabel trend={stability.trend} value={stability.value} /> : null,
                icon: "mdi:shield-check-outline",
                color: stabilityColor,
              },
              {
                label: "Active Alerts",
                value: stats ? String(stats.activeAlerts) : "—",
                sub: <span className="text-[11px] text-[#6B7280]">Requires attention</span>,
                icon: "mdi:alert-circle-outline",
                color: stats && stats.activeAlerts > 3 ? "#8C6A3D" : "#6B7280",
              },
              {
                label: "Regions Monitored",
                value: stats ? String(stats.regionsMonitored) : "—",
                sub: <span className="text-[11px] text-[#6B7280]">Active snapshots</span>,
                icon: "mdi:map-marker-multiple-outline",
                color: "#6B7280",
              },
              {
                label: "Data Sources",
                value: stats ? String(stats.dataSources) : "—",
                sub: <span className="text-[11px] text-[#6B7280]">Feature inputs</span>,
                icon: "mdi:database-outline",
                color: "#6B7280",
              },
              {
                label: "Last Update",
                value: stats ? formatTime(stats.lastUpdate) : "—",
                sub: <span className="text-[11px] text-[#6B7280]">{stats ? formatDate(stats.lastUpdate) : ""}</span>,
                icon: "mdi:clock-outline",
                color: "#6B7280",
              },
            ].map((card) => (
              <div
                key={card.label}
                className="flex items-center gap-3 bg-white border border-[#E5E7EB] rounded-[8px] px-4 py-2.5 min-w-[120px]"
              >
                <Icon icon={card.icon} className="h-5 w-5 flex-shrink-0" style={{ color: card.color }} />
                <div>
                  <div className="text-[11px] text-[#9CA3AF]">{card.label}</div>
                  <div className="text-[18px] font-semibold leading-tight text-[#111111]">
                    {card.value}
                  </div>
                  <div>{card.sub}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ═══════════════ MAP + INTEL PANEL ═══════════════ */}
      <div className="flex flex-1 overflow-hidden" style={{ minHeight: "calc(100vh - 280px)" }}>

        {/* Map — 65% */}
        <div className="relative flex flex-col flex-[65] min-w-0 border-r border-[#E5E7EB] bg-white p-5">
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="text-[11px] uppercase tracking-[0.2em] text-[#9CA3AF]">Kenya Risk Surface</div>
              <div className="text-[13px] text-[#374151] mt-0.5">
                {selectedRegion
                  ? `Viewing: ${selectedRegion.name}`
                  : "Click a region to open intelligence analysis"}
              </div>
            </div>
            {selectedRegion && (
              <div
                className="text-[12px] px-2.5 py-1 rounded-full border font-medium animate-fade-in"
                style={{
                  backgroundColor: selectedRegion.riskLevel === "critical" ? "#EDE8E0"
                    : selectedRegion.riskLevel === "elevated" ? "#F5EFE6"
                      : selectedRegion.riskLevel === "watch" ? "#FBF5EB"
                        : "#EDF4EB",
                  color: selectedRegion.riskLevel === "critical" ? "#2A1E12"
                    : selectedRegion.riskLevel === "elevated" ? "#5A3E1E"
                      : selectedRegion.riskLevel === "watch" ? "#7A4F1E"
                        : "#3A6B33",
                  borderColor: "#D1D5DB",
                }}
              >
                Score {Math.round(selectedRegion.riskScore * 100)} · {selectedRegion.confidence} confidence
              </div>
            )}
          </div>
          <div className="flex-1">
            <RiskMap
              onRegionSelect={handleRegionSelect}
              selectedRegionId={selectedRegion?.id}
            />
          </div>
        </div>

        {/* Intelligence Panel — 35% */}
        <div
          className="flex-[35] min-w-[300px] max-w-[440px] bg-white border-l border-[#E5E7EB] overflow-hidden flex flex-col"
        >
          <div className="border-b border-[#F3F4F6] px-5 py-3 flex items-center justify-between flex-shrink-0">
            <div className="text-[11px] uppercase tracking-[0.2em] text-[#9CA3AF] font-semibold">
              Intelligence Analysis
            </div>
            {selectedRegion && (
              <span className="text-[11px] text-[#9CA3AF]">auto-updates on click</span>
            )}
          </div>
          <div className="flex-1 overflow-hidden">
            <IntelligencePanel region={selectedRegion} />
          </div>
        </div>
      </div>

      {/* ═══════════════ BOTTOM ANALYTICS ═══════════════ */}
      <div className="border-t border-[#E5E7EB] bg-[#F7F7F5] px-4 py-3">
        <AnalyticsStrip
          trendData={trendData}
          comparisonData={comparisonData}
          selectedRegionName={selectedRegion?.name}
        />
      </div>
    </div>
  );
}
