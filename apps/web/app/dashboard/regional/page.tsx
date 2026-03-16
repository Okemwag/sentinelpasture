"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/dashboard/page-header";
import RiskMap from "@/components/dashboard/risk-map";
import { getRiskPalette } from "@/lib/kenya-map";
import { apiClient } from "@/lib/api-client";

type RegionRow = {
  region: string;
  population: string;
  stabilityIndex: number;
  trend: string;
  thresholdStatus: string;
  riskScore: number;
  riskLevel: "low" | "watch" | "elevated" | "critical";
  primaryDriver: string;
  storySummary: string;
  confidence: string;
};

export default function RegionalRiskPage() {
  const [rows, setRows] = useState<RegionRow[]>([]);
  const [metadata, setMetadata] = useState({ modelVersion: "—", lastUpdated: "—", confidence: "Medium" });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    apiClient.getRegionalData().then((r) => {
      if (!active) return;
      setRows(r.data);
      if (r.metadata) setMetadata(r.metadata);
    }).catch((err) => {
      if (!active) return;
      setError(err instanceof Error ? err.message : "Unable to load regional data");
    });
    return () => { active = false; };
  }, []);

  return (
    <div className="max-w-[1400px] mx-auto p-6 space-y-6">
      <PageHeader title="Regional Risk Assessment" />

      {error && (
        <div className="rounded-[8px] border border-[#E2C99A] bg-[#FBF5EB] p-3 text-[13px] text-[#7A4F1E]">
          {error}
        </div>
      )}

      {/* Map */}
      <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
        <h2 className="text-[15px] font-semibold text-[#111111] mb-4">Regional Risk Distribution</h2>
        <div style={{ height: 480 }}>
          <RiskMap />
        </div>
      </div>

      {/* Table */}
      <div className="bg-white border border-[#E5E7EB] rounded-[8px]">
        <div className="border-b border-[#F3F4F6] px-5 py-3 flex items-center justify-between">
          <h2 className="text-[15px] font-semibold text-[#111111]">Regional Summary</h2>
          <div className="flex gap-4 text-[12px] text-[#9CA3AF]">
            <span>Model: {metadata.modelVersion}</span>
            <span>Updated: {metadata.lastUpdated}</span>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[768px]">
            <thead>
              <tr className="border-b border-[#F3F4F6]">
                {["Region", "Population", "Risk Level", "Stability Index", "Primary Driver", "Why This Matters", "Confidence"].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-[11px] font-medium uppercase tracking-[0.1em] text-[#9CA3AF]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(rows.length ? rows : PLACEHOLDER_ROWS).map((region) => {
                const pal = getRiskPalette(region.riskLevel);
                return (
                  <tr
                    key={region.region}
                    className="border-b border-[#F9FAFB] last:border-0 hover:bg-[#FAFAFA] transition-colors"
                  >
                    <td className="px-4 py-3 text-[14px] font-medium text-[#111111]">{region.region}</td>
                    <td className="px-4 py-3 text-[13px] text-[#6B7280]">{region.population}</td>
                    <td className="px-4 py-3">
                      <span
                        className="inline-block text-[11px] font-semibold px-2.5 py-0.5 rounded-full border"
                        style={{ backgroundColor: pal.fill, color: pal.textColor, borderColor: pal.stroke, opacity: 0.9 }}
                      >
                        {pal.label}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="w-16 h-1.5 bg-[#F3F4F6] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${region.stabilityIndex}%`, backgroundColor: pal.fill }}
                          />
                        </div>
                        <span className="text-[13px] text-[#374151]">{region.stabilityIndex}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[13px] text-[#6B7280]">{region.primaryDriver}</td>
                    <td className="px-4 py-3 text-[12px] text-[#6B7280] leading-relaxed max-w-[240px]">{region.storySummary}</td>
                    <td className="px-4 py-3 text-[13px] text-[#374151]">{region.confidence}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

const PLACEHOLDER_ROWS: RegionRow[] = [
  { region: "North Eastern Drylands", population: "841,000", stabilityIndex: 28, trend: "Increasing", thresholdStatus: "High threshold crossed", riskScore: 0.74, riskLevel: "elevated", primaryDriver: "Rainfall anomaly", storySummary: "Flood risk affecting pastoral routes and market access.", confidence: "Medium" },
  { region: "Coast Belt", population: "1,100,000", stabilityIndex: 44, trend: "Stable", thresholdStatus: "Watch threshold", riskScore: 0.44, riskLevel: "watch", primaryDriver: "Radicalization corridors", storySummary: "Long-standing risk corridors under monitoring.", confidence: "Medium" },
  { region: "South Rift Valley", population: "630,000", stabilityIndex: 47, trend: "Stable", thresholdStatus: "Watch threshold", riskScore: 0.41, riskLevel: "watch", primaryDriver: "Livestock theft", storySummary: "Seasonal tension in pastoral zones.", confidence: "Medium" },
  { region: "Central Highlands", population: "1,200,000", stabilityIndex: 82, trend: "Stable", thresholdStatus: "Below threshold", riskScore: 0.18, riskLevel: "low", primaryDriver: "Political sensitivity", storySummary: "Stable. Monitor election-period indicators.", confidence: "High" },
  { region: "Nairobi Metro", population: "4,400,000", stabilityIndex: 78, trend: "Stable", thresholdStatus: "Below threshold", riskScore: 0.22, riskLevel: "low", primaryDriver: "Urban unrest risk", storySummary: "Stable. Cost-of-living triggers being monitored.", confidence: "High" },
];
