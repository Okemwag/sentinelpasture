"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/dashboard/page-header";
import DriverBar from "@/components/dashboard/driver-bar";
import CausalChain from "@/components/dashboard/causal-chain";
import { apiClient } from "@/lib/api-client";

type Driver = {
  label: string;
  percentage: number;
  trend: "up" | "down" | "stable";
  confidence: string;
};

const SOURCE_MAP: Record<string, string> = {
  "Rainfall anomaly": "Satellite + Climate API",
  "Vegetation index": "NDVI satellite imagery",
  "Market volatility": "Market price monitoring",
  "Mobility index": "Movement pattern analysis",
  "Incident frequency": "Security incident database",
  "Livestock movement": "Pastoral tracking data",
};

function getSource(label: string): string {
  for (const [key, val] of Object.entries(SOURCE_MAP)) {
    if (label.toLowerCase().includes(key.toLowerCase())) return val;
  }
  return "Modelled composite";
}

export default function DriversPage() {
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [metadata, setMetadata] = useState({ modelVersion: "—", lastUpdated: "—", confidence: "Medium" });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    apiClient.getDrivers().then((r) => {
      if (!active) return;
      setDrivers(r.data);
      if (r.metadata) setMetadata(r.metadata);
    }).catch((err) => {
      if (!active) return;
      setError(err instanceof Error ? err.message : "Unable to load drivers");
      setDrivers(PLACEHOLDER_DRIVERS);
    });
    return () => { active = false; };
  }, []);

  const displayed = drivers.length ? drivers : PLACEHOLDER_DRIVERS;

  return (
    <div className="max-w-[1400px] mx-auto p-6 space-y-6">
      <PageHeader title="Risk Drivers" />

      {error && (
        <div className="rounded-[8px] border border-[#E2C99A] bg-[#FBF5EB] p-3 text-[13px] text-[#7A4F1E]">
          {error}
        </div>
      )}

      {/* Meta */}
      <div className="flex flex-wrap gap-4 text-[12px] text-[#9CA3AF]">
        <span>Model: {metadata.modelVersion}</span>
        <span>Updated: {metadata.lastUpdated}</span>
        <span>Confidence: {metadata.confidence}</span>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Driver bars */}
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
          <h2 className="text-[15px] font-semibold text-[#111111] mb-5">Driver Contribution Breakdown</h2>
          <div className="space-y-5">
            {displayed.map((d) => (
              <div key={d.label}>
                <DriverBar {...d} />
                <div className="mt-1 text-[10px] text-[#9CA3AF]">
                  Source: {getSource(d.label)}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Driver definitions */}
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
          <h2 className="text-[15px] font-semibold text-[#111111] mb-5">Driver Definitions</h2>
          <div className="space-y-4">
            {displayed.map((d) => {
              const color = d.percentage >= 70 ? "#4A3A26" : d.percentage >= 50 ? "#8C6A3D" : "#C7A56B";
              return (
                <div key={d.label} className="border-l-2 pl-3" style={{ borderColor: color }}>
                  <div className="text-[13px] font-semibold text-[#111111] mb-0.5">{d.label}</div>
                  <div className="text-[12px] text-[#6B7280]">
                    Modelled contribution: <strong>{d.percentage.toFixed(1)}%</strong> with{" "}
                    {d.confidence.toLowerCase()} confidence. Data source: {getSource(d.label)}.
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Causal chain */}
      <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
        <h2 className="text-[15px] font-semibold text-[#111111] mb-4">Causal Chain — Signal Cascade</h2>
        <p className="text-[13px] text-[#6B7280] mb-5">
          How individual data signals propagate into instability pressure across the monitoring system.
        </p>
        <CausalChain />
      </div>
    </div>
  );
}

const PLACEHOLDER_DRIVERS: Driver[] = [
  { label: "Rainfall anomaly", percentage: 78, trend: "up", confidence: "High" },
  { label: "Vegetation index decline", percentage: 61, trend: "up", confidence: "High" },
  { label: "Market price volatility", percentage: 44, trend: "up", confidence: "Medium" },
  { label: "Mobility index shift", percentage: 38, trend: "stable", confidence: "Medium" },
  { label: "Incident frequency", percentage: 27, trend: "down", confidence: "Low" },
];
