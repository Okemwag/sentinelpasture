"use client";

import { useEffect, useState } from "react";

import { apiClient } from "@/lib/api-client";

type RegionRow = {
  region: string;
  stabilityIndex: number;
  trend: string;
  thresholdStatus: string;
  primaryDriver: string;
  confidence: string;
  storySummary: string;
};

export default function PressureZonesTable() {
  const [rows, setRows] = useState<RegionRow[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await apiClient.getRegionalData();
        if (!active) {
          return;
        }
        const sorted = [...response.data].sort((left, right) => left.stabilityIndex - right.stabilityIndex);
        setRows(sorted.slice(0, 6));
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load pressure zones");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return <div className="px-6 text-[13px] text-[#991B1B]">{error}</div>;
  }

  return (
    <div className="overflow-x-auto -mx-6 px-6">
      <table className="w-full min-w-[640px]">
        <thead>
          <tr className="border-b border-[#E5E7EB]">
            <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Region</th>
            <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Alert Status</th>
            <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Stability Index</th>
            <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Primary Driver</th>
            <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Why This Matters</th>
            <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {(rows.length ? rows : placeholderRows).map((zone) => (
            <tr
              key={zone.region}
              className="border-b border-[#E5E7EB] last:border-0 transition-colors duration-150 hover:bg-[#F9FAFB]"
            >
              <td className="px-4 py-3 text-[15px] text-[#111111]">{zone.region}</td>
              <td className="px-4 py-3 text-[15px] font-medium text-[#111111]">{zone.thresholdStatus || zone.trend}</td>
              <td className="px-4 py-3 text-[15px] text-[#111111]">{zone.stabilityIndex}</td>
              <td className="px-4 py-3 text-[15px] text-[#6B7280]">{zone.primaryDriver}</td>
              <td className="px-4 py-3 text-[14px] leading-6 text-[#4B5A52]">{zone.storySummary}</td>
              <td className="px-4 py-3 text-[15px] text-[#111111]">{zone.confidence}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const placeholderRows: RegionRow[] = [
  {
    region: "Loading regions",
    stabilityIndex: 0,
    trend: "Pending",
    thresholdStatus: "Pending",
    primaryDriver: "Pending",
    confidence: "Pending",
    storySummary: "Loading regional narrative",
  },
];
