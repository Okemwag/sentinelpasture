"use client";

import { useEffect, useState } from "react";

import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import RiskMap from "@/components/dashboard/risk-map";
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
  const [metadata, setMetadata] = useState({
    modelVersion: "loading",
    lastUpdated: "Loading...",
    confidence: "Medium",
  });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await apiClient.getRegionalData();
        if (!active) {
          return;
        }
        setRows(response.data);
        if (response.metadata) {
          setMetadata(response.metadata);
        }
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load regional data");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader title="Regional Risk Assessment" />

      {error ? (
        <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] p-4 text-[13px] text-[#991B1B]">
          {error}
        </div>
      ) : null}

      <Panel title="Regional Distribution">
        <RiskMap />
      </Panel>

      <Panel
        title="Regional Summary"
        metadata={{
          modelVersion: metadata.modelVersion,
          lastUpdated: metadata.lastUpdated,
          confidence: metadata.confidence,
          showExplainLink: true,
        }}
      >
        <div className="overflow-x-auto -mx-6 px-6">
          <table className="w-full min-w-[768px]">
            <thead>
              <tr className="border-b border-[#E5E7EB]">
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Region</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Population</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Alert Status</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Stability Index</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Primary Driver</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Why This Matters</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((region) => (
                <tr
                  key={region.region}
                  className="border-b border-[#E5E7EB] last:border-0 transition-colors duration-150 hover:bg-[#F9FAFB]"
                >
                  <td className="px-4 py-3 text-[15px] font-medium text-[#111111]">{region.region}</td>
                  <td className="px-4 py-3 text-[15px] text-[#6B7280]">{region.population}</td>
                  <td className="px-4 py-3 text-[15px] font-medium text-[#111111]">{region.thresholdStatus || region.trend}</td>
                  <td className="px-4 py-3 text-[15px] text-[#111111]">{region.stabilityIndex}</td>
                  <td className="px-4 py-3 text-[15px] text-[#6B7280]">{region.primaryDriver}</td>
                  <td className="px-4 py-3 text-[14px] leading-6 text-[#4B5A52]">{region.storySummary}</td>
                  <td className="px-4 py-3 text-[15px] text-[#111111]">{region.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
