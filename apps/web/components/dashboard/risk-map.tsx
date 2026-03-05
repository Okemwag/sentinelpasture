"use client";

import { useEffect, useState } from "react";

import { apiClient } from "@/lib/api-client";

type MapRegion = {
  id: string;
  name: string;
  riskLevel: "low" | "moderate" | "high";
  primaryDriver: string;
  secondaryDriver: string;
  confidence: string;
  coordinates: { x: number; y: number; width: number; height: number };
};

export default function RiskMap() {
  const [regions, setRegions] = useState<MapRegion[]>([]);
  const [hoveredRegion, setHoveredRegion] = useState<MapRegion | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await apiClient.getRegionalMap();
        if (!active) {
          return;
        }
        setRegions(response.data);
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load regional map");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  const visibleRegions = regions.length ? regions : placeholderRegions;

  return (
    <div className="relative">
      {error ? (
        <div className="mb-3 text-[13px] text-[#991B1B]">{error}</div>
      ) : null}
      <svg viewBox="0 0 400 400" className="h-[400px] w-full rounded-[8px] bg-[#F9FAFB]">
        {visibleRegions.map((region) => (
          <rect
            key={region.id}
            x={region.coordinates.x}
            y={region.coordinates.y}
            width={region.coordinates.width}
            height={region.coordinates.height}
            fill={getRiskColor(region.riskLevel)}
            stroke="#E5E7EB"
            strokeWidth="1"
            className="cursor-pointer transition-opacity duration-150 hover:opacity-80"
            onMouseEnter={() => setHoveredRegion(region)}
            onMouseLeave={() => setHoveredRegion(null)}
          />
        ))}
      </svg>

      {hoveredRegion ? (
        <div className="absolute right-4 top-4 w-64 rounded-[8px] border border-[#E5E7EB] bg-white p-4 shadow-sm">
          <div className="mb-3 text-[15px] font-medium text-[#111111]">{hoveredRegion.name}</div>
          <div className="space-y-2 text-[13px]">
            <InfoRow label="Risk Level" value={hoveredRegion.riskLevel} />
            <InfoRow label="Primary Driver" value={hoveredRegion.primaryDriver} />
            <InfoRow label="Secondary Driver" value={hoveredRegion.secondaryDriver} />
            <InfoRow label="Confidence" value={hoveredRegion.confidence} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span className="text-[#6B7280]">{label}:</span>
      <span className="capitalize text-[#111111]">{value}</span>
    </div>
  );
}

function getRiskColor(level: MapRegion["riskLevel"]) {
  if (level === "low") return "#D1D5DB";
  if (level === "moderate") return "#9CA3AF";
  return "#4B5563";
}

const placeholderRegions: MapRegion[] = [
  {
    id: "loading",
    name: "Loading",
    riskLevel: "moderate",
    primaryDriver: "Pending",
    secondaryDriver: "Pending",
    confidence: "Pending",
    coordinates: { x: 120, y: 120, width: 140, height: 90 },
  },
];
