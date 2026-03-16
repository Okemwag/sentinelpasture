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

function fmt(iso: string, mode: "time" | "date") {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return mode === "time"
    ? d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function TrendIcon({ trend }: { trend: "up" | "down" | "stable" }) {
  return trend === "up"
    ? <Icon icon="mdi:trending-up" className="h-3.5 w-3.5 text-[var(--intel-risk-elevated)]" />
    : trend === "down"
      ? <Icon icon="mdi:trending-down" className="h-3.5 w-3.5 text-[var(--intel-risk-low)]" />
      : <Icon icon="mdi:minus" className="h-3.5 w-3.5 text-[var(--intel-text-muted)]" />;
}

export default function DashboardOverview() {
  const [stats, setStats] = useState<QuickStats | null>(null);
  const [stability, setStability] = useState<StabilityData | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<MapRegion | null>(null);
  const [trendData, setTrendData] = useState<ChartPoint[]>([]);
  const [comparisonData, setComparisonData] = useState<RegionComparison[]>([]);
  const [meta, setMeta] = useState({ lastUpdated: "Loading…", modelVersion: "—" });

  useEffect(() => {
    Promise.all([apiClient.getQuickStats(), apiClient.getStabilityIndex()])
      .then(([s, stab]) => {
        setStats(s.data);
        setStability(stab.data);
        setMeta(stab.metadata ?? { lastUpdated: "—", modelVersion: "—", confidence: "Medium" });
      })
      .catch(() => { });

    apiClient.getOutcomesChart().then((r) => setTrendData(r.data)).catch(() => { });

    apiClient.getRegionalData().then((r) => {
      const top = r.data
        .slice()
        .sort((a, b) => b.riskScore - a.riskScore)
        .slice(0, 8)
        .map((d) => ({ region: d.region, riskScore: d.riskScore, riskLevel: d.riskLevel }));
      setComparisonData(top);
    }).catch(() => { });
  }, []);

  const handleRegionSelect = useCallback((region: MapRegion) => {
    setSelectedRegion(region);
  }, []);

  const score = stability?.value ?? 0;
  const scoreColor =
    score >= 70
      ? "var(--intel-risk-critical)"
      : score >= 50
        ? "var(--intel-risk-elevated)"
        : score >= 30
          ? "var(--intel-risk-watch)"
          : "var(--intel-risk-low)";

  return (
    <div
      className="flex flex-col bg-[var(--intel-bg)]"
      style={{ minHeight: "100vh" }}
    >

      {/* ══ TOP BAR ══ */}
      <div className="shrink-0 bg-[var(--intel-s0)] border-b border-[var(--intel-border)] px-5 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          {/* Left: title + subtitle */}
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold tracking-tight text-[var(--intel-text-primary)]">
                National Stability Overview
              </h1>
              <span className="hidden sm:inline-flex rounded-full bg-[var(--intel-s1)] px-2 py-0.5 text-[10px] font-medium text-[var(--intel-text-muted)]">
                Governance Early-Warning Surface
              </span>
            </div>
            <div className="flex items-center gap-3 text-[11px] text-[var(--intel-text-muted)]">
              <span className="hidden sm:inline-flex items-center gap-1">
                <span className="font-medium">{meta.modelVersion}</span>
                <span className="h-3 w-px bg-[var(--intel-border-subtle)]" />
                <span>{fmt(meta.lastUpdated, "date")}</span>
              </span>
              <span className="flex items-center gap-1.5 text-[#2F7A37]">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-live absolute inline-flex h-full w-full rounded-full bg-[#2F7A37] opacity-70" />
                  <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-[#2F7A37]" />
                </span>
                Live model feed
              </span>
            </div>
          </div>

          {/* Right: compact stat chips */}
          <div className="flex flex-wrap items-center gap-1.5">
            <StatChip
              icon="mdi:shield-check-outline"
              label="National stability index"
              value={stability ? String(Math.round(stability.value)) : "—"}
              color={scoreColor}
              emphasis
            >
              {stability && <TrendIcon trend={stability.trend} />}
            </StatChip>
            <StatChip
              icon="mdi:bell-outline"
              label="Active alerts"
              value={String(stats?.activeAlerts ?? "—")}
              color="var(--intel-risk-elevated)"
            />
            <StatChip
              icon="mdi:map-marker-multiple-outline"
              label="Regions monitored"
              value={String(stats?.regionsMonitored ?? "—")}
              color="var(--intel-text-secondary)"
            />
            <StatChip
              icon="mdi:database-outline"
              label="Data sources"
              value={String(stats?.dataSources ?? "—")}
              color="var(--intel-text-secondary)"
            />
            <StatChip
              icon="mdi:clock-outline"
              label="Last updated"
              value={stats ? fmt(stats.lastUpdate, "time") : "—"}
              color="var(--intel-text-secondary)"
            />
          </div>
        </div>
      </div>

      {/* ══ MAP + INTEL PANEL (content area) ══ */}
      <div className="flex-1">
        <div className="max-w-[1400px] mx-auto px-5 py-4">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)] items-start">
            {/* Map card */}
            <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[10px] overflow-hidden">
              {/* Map header */}
              <div className="px-5 pt-4 pb-3 space-y-2 border-b border-[var(--intel-border-subtle)]">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--intel-text-muted)]">
                      Kenya Risk Surface
                    </div>
                    <div className="text-[12px] text-[var(--intel-text-secondary)] mt-0.5">
                      {selectedRegion
                        ? `Active: ${selectedRegion.name}`
                        : "Click a region to open intelligence analysis"}
                    </div>
                  </div>
                  {selectedRegion && (
                    <div className="text-[11px] px-2.5 py-1 rounded-full border border-[var(--intel-border)] bg-[var(--intel-s1)] text-[var(--intel-text-secondary)] animate-fade-in">
                      Score {Math.round(selectedRegion.riskScore * 100)} · {selectedRegion.confidence} confidence
                    </div>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-3 text-[10px] text-[var(--intel-text-muted)]">
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: "var(--intel-risk-low)" }} />
                    Low
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: "var(--intel-risk-watch)" }} />
                    Watch
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: "var(--intel-risk-elevated)" }} />
                    Elevated
                  </span>
                  <span className="inline-flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: "var(--intel-risk-critical)" }} />
                    Critical
                  </span>
                  <span className="ml-auto text-[10px] text-[var(--intel-text-muted)] hidden sm:inline">
                    Index scaled 0–100 across monitored regions.
                  </span>
                </div>
              </div>

              {/* Map body */}
              <div className="px-5 pb-5 pt-3">
                <div className="w-full min-h-[340px]">
                  <RiskMap
                    onRegionSelect={handleRegionSelect}
                    selectedRegionId={selectedRegion?.id}
                  />
                </div>
              </div>
            </div>

            {/* Intelligence panel card */}
            <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[10px] shadow-[0_0_0_1px_rgba(15,23,42,0.02)]">
              <div className="px-5 py-3 border-b border-[var(--intel-border-subtle)] bg-[var(--intel-s1)] flex items-center justify-between">
                <span className="text-[10px] font-semibold uppercase tracking-[0.22em] text-[var(--intel-text-muted)]">
                  Intelligence Analysis
                </span>
                {selectedRegion && (
                  <span className="text-[10px] text-[var(--intel-text-muted)]">
                    updates on click
                  </span>
                )}
              </div>
              <IntelligencePanel region={selectedRegion} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatChip({
  icon,
  label,
  value,
  color,
  emphasis = false,
  children
}: {
  icon: string;
  label: string;
  value: string;
  color: string;
  emphasis?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div
      className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 bg-[var(--intel-s0)] border ${
        emphasis ? "border-[var(--intel-border)] shadow-[0_1px_2px_rgba(15,23,42,0.04)]" : "border-[var(--intel-border-subtle)]"
      }`}
    >
      <Icon icon={icon} className="h-3.5 w-3.5 shrink-0" style={{ color }} />
      <span className="text-[10px] text-[var(--intel-text-muted)]">{label}</span>
      <span className={`font-semibold text-[var(--intel-text-primary)] ${emphasis ? "text-[14px]" : "text-[13px]"}`}>
        {value}
      </span>
      {children}
    </div>
  );
}
