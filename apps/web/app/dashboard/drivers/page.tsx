"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/dashboard/page-header";
import DriverBar from "@/components/dashboard/driver-bar";
import CausalChain from "@/components/dashboard/causal-chain";
import ViolenceSummary from "@/components/dashboard/violence-summary";
import { apiClient } from "@/lib/api-client";

type Driver = {
  label: string;
  percentage: number;
  trend: "up" | "down" | "stable";
  confidence: string;
};

type DriverFamily = "climate" | "market" | "mobility" | "incidents" | "composite";

const SOURCE_MAP: Record<string, string> = {
  "Rainfall anomaly": "Satellite + Climate API",
  "Vegetation index": "NDVI satellite imagery",
  "Market volatility": "Market price monitoring",
  "Mobility index": "Movement pattern analysis",
  "Incident frequency": "Security incident database",
  "Livestock movement": "Pastoral tracking data",
};

const FAMILY_MAP: Record<DriverFamily, { label: string; color: string }> = {
  climate: { label: "Climate", color: "#4A7490" },
  market: { label: "Markets", color: "#7A5C96" },
  mobility: { label: "Mobility", color: "#2E6B5A" },
  incidents: { label: "Incidents & Violence", color: "#B0413E" },
  composite: { label: "Composite", color: "#6B7280" },
};

function getSource(label: string): string {
  for (const [key, val] of Object.entries(SOURCE_MAP)) {
    if (label.toLowerCase().includes(key.toLowerCase())) return val;
  }
  return "Modelled composite";
}

function getFamily(label: string): DriverFamily {
  const lower = label.toLowerCase();
  if (lower.includes("rain") || lower.includes("vegetation")) return "climate";
  if (lower.includes("market")) return "market";
  if (lower.includes("mobility") || lower.includes("livestock")) return "mobility";
  if (lower.includes("incident") || lower.includes("violence")) return "incidents";
  return "composite";
}

export default function DriversPage() {
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [metadata, setMetadata] = useState({ modelVersion: "—", lastUpdated: "—", confidence: "Medium" });
  const [error, setError] = useState("");
  const [violenceSeries, setViolenceSeries] = useState<Array<{
    date: string;
    totalEvents: number;
    totalFatalities: number;
    demonstrationsEvents: number;
    civilianTargetingEvents: number;
    politicalViolenceEvents: number;
  }>>([]);

  useEffect(() => {
    let active = true;
    apiClient.getDrivers()
      .then((r) => {
        if (!active) return;
        setDrivers(r.data);
        if (r.metadata) setMetadata(r.metadata);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load drivers");
      });

    apiClient.getViolenceTimeseries()
      .then((r) => {
        if (!active) return;
        setViolenceSeries(r.data);
      })
      .catch(() => {
        if (!active) return;
        setViolenceSeries([]);
      });
    return () => { active = false; };
  }, []);

  const topDrivers = [...drivers]
    .sort((a, b) => b.percentage - a.percentage)
    .slice(0, 3);
  const primary = topDrivers[0];
  const secondary = topDrivers[1];

  return (
    <div className="max-w-[1400px] mx-auto p-6 space-y-6">
      <PageHeader title="Risk Drivers" />

      {error && (
        <div className="rounded-[8px] border border-[#E2C99A] bg-[#FBF5EB] p-3 text-[13px] text-[#7A4F1E]">
          {error}
        </div>
      )}

      {/* Meta + quick takeaway */}
      <div className="space-y-2">
        <div className="flex flex-wrap gap-4 text-[12px] text-[#9CA3AF]">
          <span>Model: {metadata.modelVersion}</span>
          <span>Updated: {metadata.lastUpdated}</span>
          <span>Confidence: {metadata.confidence}</span>
        </div>
        {primary && secondary && (
          <div className="text-[13px] text-[#4B5563] bg-[#F9FAFB] border border-[#E5E7EB] rounded-[8px] px-4 py-2">
            <span className="font-semibold text-[#111827]">Current national risk is primarily driven by </span>
            <strong>{primary.label}</strong>
            <span> ({primary.percentage.toFixed(1)}%) and </span>
            <strong>{secondary.label}</strong>
            <span>. Conflict and violence signals contribute via incident-related drivers and the monthly violence series.</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Driver bars */}
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
          <h2 className="text-[15px] font-semibold text-[#111111] mb-5">Driver Contribution Breakdown</h2>
          {drivers.length === 0 ? (
            <div className="text-[13px] text-[#9CA3AF]">
              No driver contributions are currently available from the model.
            </div>
          ) : (
            <div className="space-y-5">
              {drivers.map((d) => {
                const family = getFamily(d.label);
                const famMeta = FAMILY_MAP[family];
                return (
                  <div key={d.label}>
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="inline-flex items-center gap-1.5 rounded-full bg-[#F3F4F6] px-2 py-0.5">
                        <span
                          className="h-1.5 w-1.5 rounded-full"
                          style={{ backgroundColor: famMeta.color }}
                        />
                        <span className="text-[10px] font-medium text-[#4B5563]">
                          {famMeta.label}
                        </span>
                      </div>
                      <span className="text-[11px] text-[#9CA3AF]">
                        {d.percentage.toFixed(1)}%
                      </span>
                    </div>
                    <DriverBar {...d} />
                    <div className="mt-1 text-[10px] text-[#9CA3AF]">
                      Source: {getSource(d.label)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Driver definitions */}
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
          <h2 className="text-[15px] font-semibold text-[#111111] mb-5">Driver Definitions</h2>
          {drivers.length === 0 ? (
            <div className="text-[13px] text-[#9CA3AF]">
              Once driver signals are available, their definitions and contributions will appear here.
            </div>
          ) : (
            <div className="space-y-4">
              {drivers.map((d) => {
                const family = getFamily(d.label);
                const famMeta = FAMILY_MAP[family];
                const color = famMeta.color;
                const isIncident = family === "incidents";
                return (
                  <div key={d.label} className="border-l-2 pl-3" style={{ borderColor: color }}>
                    <div className="flex items-center gap-2 mb-0.5">
                      <div className="inline-flex items-center gap-1.5 rounded-full bg-[#F3F4F6] px-2 py-0.5">
                        <span
                          className="h-1.5 w-1.5 rounded-full"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-[10px] font-medium text-[#4B5563]">
                          {famMeta.label}
                        </span>
                      </div>
                      <div className="text-[13px] font-semibold text-[#111111]">
                        {d.label}
                      </div>
                    </div>
                    <div className="text-[12px] text-[#6B7280]">
                      Modelled contribution: <strong>{d.percentage.toFixed(1)}%</strong> with{" "}
                      {d.confidence.toLowerCase()} confidence. Data source: {getSource(d.label)}.
                      {isIncident && (
                        <> This driver is informed by monthly demonstrations and political violence events shown in the conflict & violence panel.</>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Causal chain */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
          <h2 className="text-[15px] font-semibold text-[#111111] mb-4">Causal Chain — Signal Cascade</h2>
          <p className="text-[13px] text-[#6B7280] mb-5">
            How individual data signals propagate into instability pressure across the monitoring system.
          </p>
          <CausalChain />
        </div>
        <ViolenceSummary series={violenceSeries} />
      </div>
    </div>
  );
}
// No local placeholder drivers; UI shows explicit empty states when none are available.
