"use client";

import { useState } from "react";

interface Region {
  id: string;
  name: string;
  riskLevel: "low" | "moderate" | "high";
  primaryDriver: string;
  secondaryDriver: string;
  confidence: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

const mockRegions: Region[] = [
  {
    id: "north",
    name: "Northern Region",
    riskLevel: "moderate",
    primaryDriver: "Economic stress",
    secondaryDriver: "Climate anomaly",
    confidence: "High",
    x: 150,
    y: 50,
    width: 120,
    height: 80,
  },
  {
    id: "central",
    name: "Central Region",
    riskLevel: "low",
    primaryDriver: "Mobility disruption",
    secondaryDriver: "Education decline",
    confidence: "Medium",
    x: 140,
    y: 150,
    width: 140,
    height: 100,
  },
  {
    id: "south",
    name: "Southern Region",
    riskLevel: "high",
    primaryDriver: "Incident contagion",
    secondaryDriver: "Economic stress",
    confidence: "High",
    x: 160,
    y: 270,
    width: 100,
    height: 90,
  },
];

export default function RiskMap() {
  const [hoveredRegion, setHoveredRegion] = useState<Region | null>(null);

  const getRiskColor = (level: string) => {
    if (level === "low") return "#D1D5DB";
    if (level === "moderate") return "#9CA3AF";
    return "#4B5563";
  };

  return (
    <div className="relative">
      <svg
        viewBox="0 0 400 400"
        className="w-full h-[400px] bg-[#F9FAFB] rounded-[8px]"
      >
        {mockRegions.map((region) => (
          <rect
            key={region.id}
            x={region.x}
            y={region.y}
            width={region.width}
            height={region.height}
            fill={getRiskColor(region.riskLevel)}
            stroke="#E5E7EB"
            strokeWidth="1"
            className="cursor-pointer transition-opacity duration-150 hover:opacity-80"
            onMouseEnter={() => setHoveredRegion(region)}
            onMouseLeave={() => setHoveredRegion(null)}
          />
        ))}
      </svg>

      {hoveredRegion && (
        <div className="absolute top-4 right-4 bg-white border border-[#E5E7EB] rounded-[8px] p-4 shadow-sm w-64">
          <div className="text-[15px] font-medium text-[#111111] mb-3">
            {hoveredRegion.name}
          </div>
          <div className="space-y-2 text-[13px]">
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Risk Level:</span>
              <span className="text-[#111111] capitalize">
                {hoveredRegion.riskLevel}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Primary Driver:</span>
              <span className="text-[#111111]">{hoveredRegion.primaryDriver}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Secondary Driver:</span>
              <span className="text-[#111111]">{hoveredRegion.secondaryDriver}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#6B7280]">Confidence:</span>
              <span className="text-[#111111]">{hoveredRegion.confidence}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
