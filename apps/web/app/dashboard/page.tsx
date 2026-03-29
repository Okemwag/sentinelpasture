"use client";

import { useEffect, useState, useCallback } from "react";
import { Icon } from "@iconify/react";

import RiskMap, { type MapRegion, type MapEventPoint } from "@/components/dashboard/risk-map";
import IntelligencePanel from "@/components/dashboard/intelligence-panel";
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
type AlertRow = {
  id: number;
  severity: "elevated" | "moderate";
  title: string;
  description: string;
  timestamp: string;
  region: string;
  status: "active" | "monitoring" | "resolved";
};
type DriverRow = {
  label: string;
  percentage: number;
  trend: "up" | "down" | "stable";
  confidence: string;
};
type ViolencePoint = {
  date: string;
  totalEvents: number;
  totalFatalities: number;
  demonstrationsEvents: number;
  civilianTargetingEvents: number;
  politicalViolenceEvents: number;
};
type InterventionRow = {
  category: string;
  expectedImpact: string;
  timeToEffect: string;
  costBand: string;
  confidence: string;
};

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
  const [meta, setMeta] = useState({ lastUpdated: "Loading…", modelVersion: "—" });
  const [alerts, setAlerts] = useState<AlertRow[]>([]);
  const [drivers, setDrivers] = useState<DriverRow[]>([]);
  const [violenceSeries, setViolenceSeries] = useState<ViolencePoint[]>([]);
  const [activeLayer, setActiveLayer] = useState<"risk" | "events" | "mobility" | "satellite">("risk");
  const [timeWindow, setTimeWindow] = useState<"7d" | "30d" | "90d">("30d");
  const [satelliteTimestamp, setSatelliteTimestamp] = useState("latest");
  const [satelliteCompare, setSatelliteCompare] = useState(50);
  const [playbackEnabled, setPlaybackEnabled] = useState(false);
  const [playbackProgress, setPlaybackProgress] = useState(100);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2 | 4>(1);
  const [selectedEvent, setSelectedEvent] = useState<MapEventPoint | null>(null);
  const [satelliteDates, setSatelliteDates] = useState<string[]>([]);
  const [satelliteTileTemplate, setSatelliteTileTemplate] = useState<string | undefined>(undefined);
  const [eventInterventions, setEventInterventions] = useState<InterventionRow[]>([]);
  const [eventInterventionsLoading, setEventInterventionsLoading] = useState(false);
  const [eventInterventionsError, setEventInterventionsError] = useState("");

  useEffect(() => {
    Promise.all([apiClient.getQuickStats(), apiClient.getStabilityIndex()])
      .then(([s, stab]) => {
        setStats(s.data);
        setStability(stab.data);
        setMeta(stab.metadata ?? { lastUpdated: "—", modelVersion: "—", confidence: "Medium" });
      })
      .catch(() => { });

    apiClient.getAlerts().then((r) => setAlerts(r.data)).catch(() => { });
    apiClient.getDrivers().then((r) => setDrivers(r.data)).catch(() => { });
    apiClient.getViolenceTimeseries().then((r) => setViolenceSeries(r.data)).catch(() => { });
    apiClient.getSatelliteDates(30)
      .then((r) => {
        setSatelliteDates(r.data.dates ?? []);
        if (r.data.dates?.length) setSatelliteTimestamp(r.data.dates[0]);
      })
      .catch(() => { });
  }, []);

  useEffect(() => {
    if (activeLayer !== "satellite") return;
    apiClient.getSatelliteTileTemplate(satelliteTimestamp === "latest" ? undefined : satelliteTimestamp)
      .then((r) => setSatelliteTileTemplate(r.data.tileTemplate))
      .catch(() => setSatelliteTileTemplate(undefined));
  }, [activeLayer, satelliteTimestamp]);

  useEffect(() => {
    if (!playbackEnabled || activeLayer !== "events") return;
    const interval = window.setInterval(() => {
      setPlaybackProgress((v) => {
        if (v >= 100) return 0;
        return Math.min(100, v + playbackSpeed);
      });
    }, 500);
    return () => window.clearInterval(interval);
  }, [playbackEnabled, activeLayer, playbackSpeed]);

  const handleRegionSelect = useCallback((region: MapRegion) => {
    setSelectedRegion(region);
    // Keep drill-down actionable even when user clicks region (not an event point).
    setSelectedEvent({
      id: -1,
      title: `Regional posture update — ${region.name}`,
      description: region.thresholdReason || region.storySummary,
      timestamp: new Date().toLocaleString(),
      region: region.name,
      severity: region.riskLevel === "critical" || region.riskLevel === "elevated" ? "elevated" : "moderate",
      status: region.riskLevel === "low" ? "monitoring" : "active",
      sourceRegionId: region.sourceRegionId,
    });
  }, []);
  const handleEventSelect = useCallback((eventPoint: MapEventPoint) => {
    setSelectedEvent(eventPoint);
  }, []);

  useEffect(() => {
    if (!selectedEvent?.sourceRegionId) {
      setEventInterventions([]);
      setEventInterventionsError("");
      setEventInterventionsLoading(false);
      return;
    }
    setEventInterventionsLoading(true);
    setEventInterventionsError("");
    apiClient.getInterventions(selectedEvent.sourceRegionId)
      .then((r) => setEventInterventions(r.data))
      .catch((err) => {
        setEventInterventions([]);
        setEventInterventionsError(err instanceof Error ? err.message : "Unable to load interventions");
      })
      .finally(() => setEventInterventionsLoading(false));
  }, [selectedEvent?.sourceRegionId]);

  const score = stability?.value ?? 0;
  const scoreColor =
    score >= 70
      ? "var(--intel-risk-critical)"
      : score >= 50
        ? "var(--intel-risk-elevated)"
        : score >= 30
          ? "var(--intel-risk-watch)"
          : "var(--intel-risk-low)";
  const topDriver = [...drivers].sort((a, b) => b.percentage - a.percentage)[0];
  const latestViolence = violenceSeries[violenceSeries.length - 1];
  const days = timeWindow === "7d" ? 7 : timeWindow === "30d" ? 30 : 90;
  const cutoffMs = Date.now() - days * 24 * 60 * 60 * 1000;
  const actionQueue = alerts
    .filter((a) => Number(new Date(a.timestamp)) >= cutoffMs)
    .filter((a) => a.status !== "resolved")
    .sort((a, b) => Number(new Date(b.timestamp)) - Number(new Date(a.timestamp)))
    .slice(0, 5);
  const timelineBuckets = buildEventTimeline(alerts, timeWindow, playbackProgress);
  const topChangingRegions = buildTopChangingRegions(alerts, timeWindow, playbackProgress).slice(0, 3);

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
          {/* Phase 1: Global threat posture strip */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 mb-4">
            <PostureCard
              icon="mdi:gauge"
              label="National Stability"
              value={stability ? String(Math.round(stability.value)) : "—"}
              tone={scoreColor}
              caption={stability ? `${stability.trend} trend` : "loading"}
            />
            <PostureCard
              icon="mdi:alert-circle-outline"
              label="Active Threat Queue"
              value={String(alerts.filter((a) => a.status === "active").length)}
              tone="var(--intel-risk-elevated)"
              caption="alerts requiring action"
            />
            <PostureCard
              icon="mdi:chart-timeline-variant"
              label="Dominant Driver"
              value={topDriver ? `${topDriver.label}` : "—"}
              tone="#4A7490"
              caption={topDriver ? `${topDriver.percentage.toFixed(1)}% contribution` : "loading"}
            />
            <PostureCard
              icon="mdi:shield-alert-outline"
              label="Conflict Signal"
              value={latestViolence ? String(latestViolence.totalEvents) : "—"}
              tone="#B0413E"
              caption={latestViolence ? `${latestViolence.totalFatalities} fatalities (${latestViolence.date})` : "loading"}
            />
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)] items-start">
            {/* Map card */}
            <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[10px] overflow-hidden lg:sticky lg:top-4">
              {/* Map header */}
              <div className="px-5 pt-4 pb-3 space-y-2 border-b border-[var(--intel-border-subtle)]">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-1.5">
                    {[
                      { id: "risk", label: "Risk Layer" },
                      { id: "events", label: "Event Layer" },
                      { id: "mobility", label: "Mobility Layer" },
                      { id: "satellite", label: "Satellite Layer" },
                    ].map((layer) => (
                      <button
                        key={layer.id}
                        type="button"
                        onClick={() => setActiveLayer(layer.id as typeof activeLayer)}
                        className={`text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                          activeLayer === layer.id
                            ? "bg-[var(--intel-s2)] border-[var(--intel-border)] text-[var(--intel-text-primary)]"
                            : "bg-[var(--intel-s0)] border-[var(--intel-border-subtle)] text-[var(--intel-text-muted)] hover:bg-[var(--intel-s1)]"
                        }`}
                      >
                        {layer.label}
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center gap-1.5">
                    {(["7d", "30d", "90d"] as const).map((w) => (
                      <button
                        key={w}
                        type="button"
                        onClick={() => setTimeWindow(w)}
                        className={`text-[11px] px-2 py-1 rounded border ${
                          timeWindow === w
                            ? "bg-[var(--intel-s2)] border-[var(--intel-border)] text-[var(--intel-text-primary)]"
                            : "bg-[var(--intel-s0)] border-[var(--intel-border-subtle)] text-[var(--intel-text-muted)]"
                        }`}
                      >
                        {w}
                      </button>
                    ))}
                    {activeLayer === "events" && (
                      <div className="ml-2 flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => setPlaybackEnabled((v) => !v)}
                          className="text-[11px] px-2 py-1 rounded border border-[var(--intel-border)] bg-[var(--intel-s0)] text-[var(--intel-text-secondary)]"
                        >
                          {playbackEnabled ? "Pause" : "Play"}
                        </button>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={1}
                          value={playbackProgress}
                          onChange={(e) => setPlaybackProgress(Number(e.target.value))}
                          className="w-20 accent-[#B91C1C]"
                          aria-label="Event playback progress"
                        />
                        <select
                          value={String(playbackSpeed)}
                          onChange={(e) => setPlaybackSpeed(Number(e.target.value) as 1 | 2 | 4)}
                          className="text-[11px] border border-[var(--intel-border)] bg-[var(--intel-s0)] text-[var(--intel-text-secondary)] rounded px-1 py-1"
                        >
                          <option value="1">1x</option>
                          <option value="2">2x</option>
                          <option value="4">4x</option>
                        </select>
                      </div>
                    )}
                    {activeLayer === "satellite" && (
                      <div className="ml-2 flex items-center gap-1.5">
                        <select
                          value={satelliteTimestamp}
                          onChange={(e) => setSatelliteTimestamp(e.target.value)}
                          className="text-[11px] border border-[var(--intel-border)] bg-[var(--intel-s0)] text-[var(--intel-text-secondary)] rounded px-1.5 py-1"
                        >
                          {satelliteDates.length === 0 && <option value="latest">latest</option>}
                          {satelliteDates.map((d) => (
                            <option key={d} value={d}>{d}</option>
                          ))}
                        </select>
                        <input
                          type="range"
                          min={10}
                          max={90}
                          step={1}
                          value={satelliteCompare}
                          onChange={(e) => setSatelliteCompare(Number(e.target.value))}
                          className="w-20 accent-[#2563EB]"
                          aria-label="Satellite compare slider"
                        />
                      </div>
                    )}
                  </div>
                </div>
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
                <div className="text-[11px] text-[var(--intel-text-muted)]">
                  {activeLayer === "risk" && "Composite risk layer (climate + incidents + market + mobility)."}
                  {activeLayer === "events" && "Event emphasis layer (demonstrations, civilian targeting, political violence)."}
                  {activeLayer === "mobility" && "Mobility layer scaffold (movement pressure and corridor stress)."}
                  {activeLayer === "satellite" && "Satellite layer scaffold enabled (imagery tiles integration is next phase)."}
                </div>
                {activeLayer === "events" && (
                  <div className="text-[11px] text-[var(--intel-text-muted)]">
                    Playback: {playbackProgress}% through selected {timeWindow} window.
                  </div>
                )}
                {activeLayer === "events" && (
                  <div className="pt-1">
                    <div className="flex items-end gap-1 h-12">
                      {timelineBuckets.map((bucket) => (
                        <div key={bucket.label} className="flex-1 flex flex-col items-center gap-1">
                          <div
                            className="w-full rounded-t-[2px]"
                            style={{
                              height: `${Math.max(2, (bucket.count / Math.max(...timelineBuckets.map((b) => b.count), 1)) * 36)}px`,
                              backgroundColor: "#B91C1C",
                              opacity: bucket.played ? 0.9 : 0.28,
                            }}
                          />
                          <span className="text-[9px] text-[var(--intel-text-muted)]">{bucket.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Map body */}
              <div className="px-5 pb-5 pt-3">
                <div className="w-full min-h-[520px]">
                  <RiskMap
                    onRegionSelect={handleRegionSelect}
                    selectedRegionId={selectedRegion?.id}
                    activeLayer={activeLayer}
                    timeWindow={timeWindow}
                    satelliteTimestamp={satelliteTimestamp}
                    satelliteCompare={satelliteCompare}
                    alerts={alerts}
                    playbackProgress={playbackProgress}
                    onEventSelect={handleEventSelect}
                    satelliteTileTemplate={satelliteTileTemplate}
                  />
                </div>
                {activeLayer === "satellite" && (
                  <div className="mt-2 rounded-[8px] border border-[#C7D2FE] bg-[#EEF2FF] px-3 py-2 text-[12px] text-[#3730A3]">
                    Satellite mode is using NASA GIBS true-color raster imagery (timestamp selectable).
                  </div>
                )}
              </div>
            </div>

            {/* Analysis + actions */}
            <div className="space-y-4">
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

              <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[10px] overflow-hidden">
                <div className="px-5 py-3 border-b border-[var(--intel-border-subtle)] bg-[var(--intel-s1)]">
                  <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--intel-text-muted)]">
                    Event/Region Drill-Down
                  </h3>
                </div>
                <div className="px-5 py-4 space-y-2">
                  {selectedEvent ? (
                    <>
                      <div className="text-[13px] font-semibold text-[var(--intel-text-primary)]">
                        {selectedEvent.title}
                      </div>
                      <div className="text-[12px] text-[var(--intel-text-secondary)]">
                        {selectedEvent.region} · {selectedEvent.timestamp}
                      </div>
                      <div className="text-[12px] text-[var(--intel-text-secondary)]">
                        {selectedEvent.description}
                      </div>
                      <div className="pt-1 text-[11px] text-[var(--intel-text-muted)]">
                        Source evidence: alert feed + modelled regional pressure.
                      </div>
                      <div className="pt-1">
                        <div className="text-[11px] uppercase tracking-[0.12em] text-[var(--intel-text-muted)] mb-1">
                          Key drivers
                        </div>
                        <div className="space-y-1">
                          {drivers.slice(0, 3).map((d) => (
                            <div key={d.label} className="flex items-center justify-between text-[12px]">
                              <span className="text-[var(--intel-text-secondary)]">{d.label}</span>
                              <span className="font-semibold text-[var(--intel-text-primary)]">{d.percentage.toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="pt-1 text-[11px] text-[var(--intel-text-muted)]">
                        Recommended interventions for this exact region:
                      </div>
                      {eventInterventionsLoading && (
                        <div className="text-[12px] text-[var(--intel-text-muted)]">
                          Loading regional interventions...
                        </div>
                      )}
                      {eventInterventionsError && (
                        <div className="text-[12px] text-[#B45309]">
                          {eventInterventionsError}
                        </div>
                      )}
                      {!eventInterventionsLoading && !eventInterventionsError && eventInterventions.length > 0 && (
                        <div className="space-y-1.5">
                          {eventInterventions.slice(0, 3).map((item) => (
                            <div key={`${item.category}-${item.timeToEffect}`} className="rounded-[6px] border border-[var(--intel-border-subtle)] bg-[var(--intel-s1)] px-2.5 py-2">
                              <div className="text-[12px] font-medium text-[var(--intel-text-primary)]">{item.category}</div>
                              <div className="text-[11px] text-[var(--intel-text-secondary)]">
                                {item.expectedImpact} · {item.timeToEffect} · {item.costBand} · {item.confidence}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="text-[12px] text-[var(--intel-text-muted)]">
                      Click an event point or region to open source evidence, drivers, and recommended action context.
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[10px] overflow-hidden">
                <div className="px-5 py-3 border-b border-[var(--intel-border-subtle)] bg-[var(--intel-s1)]">
                  <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--intel-text-muted)]">
                    Action Queue
                  </h3>
                </div>
                <div className="divide-y divide-[var(--intel-border-subtle)]">
                  {actionQueue.length === 0 ? (
                    <div className="px-5 py-4 text-[12px] text-[var(--intel-text-muted)]">
                      No active operational tasks at this time.
                    </div>
                  ) : (
                    actionQueue.map((a) => (
                      <div key={a.id} className="px-5 py-3">
                        <div className="flex items-center justify-between gap-2">
                          <div className="text-[13px] font-medium text-[var(--intel-text-primary)]">
                            {a.title}
                          </div>
                          <span className="text-[10px] text-[var(--intel-text-muted)]">
                            {a.region}
                          </span>
                        </div>
                        <div className="mt-1 text-[12px] text-[var(--intel-text-secondary)] line-clamp-2">
                          {a.description}
                        </div>
                        <div className="mt-1 text-[10px] text-[var(--intel-text-muted)]">
                          {new Date(a.timestamp).toLocaleString()}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[10px] overflow-hidden">
                <div className="px-5 py-3 border-b border-[var(--intel-border-subtle)] bg-[var(--intel-s1)]">
                  <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--intel-text-muted)]">
                    Multi-Risk Composition
                  </h3>
                </div>
                <div className="px-5 py-4 space-y-3">
                  {drivers.slice(0, 4).map((driver) => (
                    <div key={driver.label}>
                      <div className="flex items-center justify-between text-[12px]">
                        <span className="text-[var(--intel-text-secondary)]">{driver.label}</span>
                        <span className="font-semibold text-[var(--intel-text-primary)]">
                          {driver.percentage.toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-1 h-1.5 bg-[var(--intel-s2)] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.min(100, driver.percentage)}%`,
                            backgroundColor:
                              driver.label.toLowerCase().includes("incident") ||
                                driver.label.toLowerCase().includes("violence")
                                ? "#B0413E"
                                : driver.label.toLowerCase().includes("market")
                                  ? "#7A5C96"
                                  : driver.label.toLowerCase().includes("mobility")
                                    ? "#2E6B5A"
                                    : "#4A7490",
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  {drivers.length === 0 && (
                    <div className="text-[12px] text-[var(--intel-text-muted)]">
                      Driver contributions are loading.
                    </div>
                  )}
                </div>
              </div>

              {activeLayer === "events" && topChangingRegions.length > 0 && (
                <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[10px] overflow-hidden">
                  <div className="px-5 py-3 border-b border-[var(--intel-border-subtle)] bg-[var(--intel-s1)]">
                    <h3 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--intel-text-muted)]">
                      Top Changing Regions
                    </h3>
                  </div>
                  <div className="px-5 py-3 space-y-2">
                    {topChangingRegions.map((item) => (
                      <div key={item.region} className="flex items-center justify-between text-[12px]">
                        <span className="text-[var(--intel-text-secondary)]">{item.region}</span>
                        <span className="font-semibold text-[#B91C1C]">+{item.delta}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function buildEventTimeline(alerts: AlertRow[], window: "7d" | "30d" | "90d", playbackProgress: number) {
  const days = window === "7d" ? 7 : window === "30d" ? 30 : 90;
  const buckets = 10;
  const now = Date.now();
  const start = now - days * 24 * 60 * 60 * 1000;
  const span = now - start;
  const playbackTs = start + (Math.max(0, Math.min(100, playbackProgress)) / 100) * span;
  return Array.from({ length: buckets }, (_, idx) => {
    const bucketStart = start + (idx / buckets) * span;
    const bucketEnd = start + ((idx + 1) / buckets) * span;
    const count = alerts.filter((a) => {
      const ts = Number(new Date(a.timestamp));
      return ts >= bucketStart && ts < bucketEnd;
    }).length;
    const played = bucketEnd <= playbackTs;
    return { label: `${idx + 1}`, count, played };
  });
}

function buildTopChangingRegions(alerts: AlertRow[], window: "7d" | "30d" | "90d", playbackProgress: number) {
  const days = window === "7d" ? 7 : window === "30d" ? 30 : 90;
  const now = Date.now();
  const start = now - days * 24 * 60 * 60 * 1000;
  const span = now - start;
  const playbackTs = start + (Math.max(0, Math.min(100, playbackProgress)) / 100) * span;
  const mid = start + span * 0.5;
  const current = new Map<string, number>();
  const previous = new Map<string, number>();
  alerts.forEach((a) => {
    const ts = Number(new Date(a.timestamp));
    if (ts < start || ts > playbackTs) return;
    if (ts >= mid) {
      current.set(a.region, (current.get(a.region) ?? 0) + 1);
    } else {
      previous.set(a.region, (previous.get(a.region) ?? 0) + 1);
    }
  });
  const allRegions = new Set([...Array.from(current.keys()), ...Array.from(previous.keys())]);
  return Array.from(allRegions)
    .map((region) => ({
      region,
      delta: (current.get(region) ?? 0) - (previous.get(region) ?? 0),
    }))
    .sort((a, b) => b.delta - a.delta);
}

function PostureCard({
  icon,
  label,
  value,
  tone,
  caption,
}: {
  icon: string;
  label: string;
  value: string;
  tone: string;
  caption: string;
}) {
  return (
    <div className="rounded-[10px] border border-[var(--intel-border)] bg-[var(--intel-s0)] px-4 py-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-[0.14em] text-[var(--intel-text-muted)]">
          {label}
        </span>
        <Icon icon={icon} className="h-4 w-4" style={{ color: tone }} />
      </div>
      <div className="mt-1 text-[18px] font-semibold text-[var(--intel-text-primary)] truncate">
        {value}
      </div>
      <div className="mt-0.5 text-[11px] text-[var(--intel-text-muted)]">{caption}</div>
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
