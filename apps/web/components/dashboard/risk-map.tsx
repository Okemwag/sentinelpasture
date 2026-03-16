"use client";

import { useEffect, useRef, useState } from "react";
import { apiClient } from "@/lib/api-client";
import {
  KENYA_OUTLINE_PATH,
  KENYA_REGION_SHAPES,
  getRiskPalette,
  type RegionalRiskLevel,
} from "@/lib/kenya-map";

export type MapRegion = {
  id: string;
  name: string;
  shapeId: string;
  riskLevel: RegionalRiskLevel;
  riskScore: number;
  thresholdStatus: string;
  primaryDriver: string;
  secondaryDriver: string;
  confidence: string;
  storySummary: string;
  thresholdReason: string;
  communityContext: string;
  watchItems: string[];
  sourceRegionId: string;
  population?: string;
};

interface RiskMapProps {
  onRegionSelect?: (region: MapRegion) => void;
  selectedRegionId?: string | null;
}

export default function RiskMap({ onRegionSelect, selectedRegionId }: RiskMapProps) {
  const [regions, setRegions] = useState<MapRegion[]>([]);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const hasAutoSelected = useRef(false);

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await apiClient.getRegionalMap();
        if (!active) return;
        const sorted = [...response.data].sort((a, b) => b.riskScore - a.riskScore);
        setRegions(sorted);
        if (!hasAutoSelected.current && sorted[0]) {
          hasAutoSelected.current = true;
          onRegionSelect?.(sorted[0]);
        }
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Unable to load regional map");
        // Load placeholders
        setRegions(PLACEHOLDER_REGIONS);
        if (!hasAutoSelected.current && PLACEHOLDER_REGIONS[0]) {
          hasAutoSelected.current = true;
          onRegionSelect?.(PLACEHOLDER_REGIONS[0]);
        }
      }
    }
    void load();
    return () => { active = false; };
  }, []);

  const visibleRegions = regions.length ? regions : PLACEHOLDER_REGIONS;
  const activeId = selectedRegionId ?? visibleRegions[0]?.id;

  // Find highest risk region for pulse ring
  const criticalRegion = visibleRegions.find((r) => r.riskLevel === "critical")
    ?? visibleRegions.find((r) => r.riskLevel === "elevated")
    ?? visibleRegions[0];

  return (
    <div className="relative h-full w-full flex flex-col">
      {error && (
        <div className="mb-2 text-[12px] text-[#8C6A3D] bg-[#FBF5EB] border border-[#E2C99A] rounded-[6px] px-3 py-2">
          Live data unavailable — showing modelled baseline
        </div>
      )}

      {/* Legend */}
      <div className="flex flex-wrap gap-2 mb-3">
        {(["low", "watch", "elevated", "critical"] as const).map((level, idx) => {
          const pal = getRiskPalette(level);
          const labels = ["Low", "Elevated", "High", "Critical"];
          return (
            <div
              key={level}
              className="inline-flex items-center gap-1.5 text-[11px] text-[#374151]"
            >
              <span
                className="inline-block h-3 w-3 rounded-sm flex-shrink-0"
                style={{ backgroundColor: pal.fill, border: `1px solid ${pal.stroke}` }}
              />
              {labels[idx]}
            </div>
          );
        })}
      </div>

      {/* Map SVG */}
      <div className="flex-1 relative">
        <svg
          viewBox="70 20 280 520"
          className="h-full w-full"
          style={{ minHeight: 400 }}
        >
          {/* Country outline shadow */}
          <path
            d={KENYA_OUTLINE_PATH}
            fill="none"
            stroke="#C8D8C2"
            strokeWidth="14"
            strokeLinejoin="round"
            strokeOpacity={0.4}
          />
          {/* Country outline */}
          <path
            d={KENYA_OUTLINE_PATH}
            fill="#F7F7F5"
            stroke="#9CA3AF"
            strokeWidth="3"
            strokeLinejoin="round"
          />

          {visibleRegions.map((region) => {
            const shape = KENYA_REGION_SHAPES[region.shapeId];
            if (!shape) return null;
            const palette = getRiskPalette(region.riskLevel);
            const isActive = activeId === region.id;
            const isHovered = hoveredId === region.id;
            const isPulse = criticalRegion?.id === region.id;

            // Compute centroid from polygon points for the pulse dot
            const pts = shape.points.split(" ").map((p) => {
              const [x, y] = p.split(",").map(Number);
              return { x, y };
            });
            const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
            const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;

            return (
              <g
                key={region.id}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredId(region.id)}
                onMouseLeave={() => setHoveredId(null)}
                onClick={() => onRegionSelect?.(region)}
              >
                <polygon
                  points={shape.points}
                  fill={palette.fill}
                  stroke={isActive ? "#111111" : isHovered ? "#374151" : palette.stroke}
                  strokeWidth={isActive ? 3 : isHovered ? 2 : 1.5}
                  opacity={isActive ? 1 : isHovered ? 0.95 : 0.85}
                  style={{ transition: "all 0.2s ease" }}
                />

                {/* Region abbreviation label */}
                <text
                  x={shape.labelX}
                  y={shape.labelY}
                  textAnchor="middle"
                  fontSize={isActive ? 10 : 9}
                  fontWeight={isActive ? 700 : 500}
                  fill={isActive ? "#111111" : "#374151"}
                  style={{ pointerEvents: "none", letterSpacing: "0.08em" }}
                >
                  {shape.label}
                </text>

                {/* Pulse ring for the highest-risk region */}
                {isPulse && (
                  <>
                    <circle
                      cx={cx}
                      cy={cy}
                      r={10}
                      fill={palette.fill}
                      fillOpacity={0.3}
                      style={{
                        transformOrigin: `${cx}px ${cy}px`,
                        animation: "pulseRing 2s ease-out infinite",
                        pointerEvents: "none",
                      }}
                    />
                    <circle cx={cx} cy={cy} r={4} fill={palette.stroke} />
                  </>
                )}
              </g>
            );
          })}
        </svg>

        {/* Hover tooltip */}
        {hoveredId && (() => {
          const r = visibleRegions.find((x) => x.id === hoveredId);
          if (!r) return null;
          const pal = getRiskPalette(r.riskLevel);
          return (
            <div
              className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-white border border-[#E5E7EB] rounded-[8px] px-3 py-2 shadow-lg text-[12px] pointer-events-none z-10 whitespace-nowrap"
              style={{ boxShadow: "0 4px 16px rgba(0,0,0,0.10)" }}
            >
              <span className="font-semibold text-[#111111]">{r.name}</span>
              <span
                className="ml-2 px-1.5 py-0.5 rounded-full text-[10px] font-semibold border"
                style={{ backgroundColor: pal.fill, color: pal.textColor, borderColor: pal.stroke }}
              >
                {pal.label}
              </span>
              <span className="ml-2 text-[#6B7280]">
                Score: {Math.round(r.riskScore * 100)}
              </span>
            </div>
          );
        })()}
      </div>

      {/* Bottom risk-level bar */}
      <div className="mt-3 flex gap-1 h-1.5 rounded-full overflow-hidden">
        {visibleRegions.map((r) => {
          const pal = getRiskPalette(r.riskLevel);
          return (
            <div
              key={r.id}
              className="flex-1 cursor-pointer transition-opacity"
              style={{ backgroundColor: pal.fill, opacity: activeId === r.id ? 1 : 0.5 }}
              onClick={() => onRegionSelect?.(r)}
              title={r.name}
            />
          );
        })}
      </div>
    </div>
  );
}

const PLACEHOLDER_REGIONS: MapRegion[] = [
  {
    id: "north_eastern_drylands",
    shapeId: "north_eastern_drylands",
    name: "North Eastern Drylands",
    riskLevel: "elevated",
    riskScore: 0.74,
    thresholdStatus: "High threshold crossed",
    primaryDriver: "Rainfall anomaly +210%",
    secondaryDriver: "Pasture shocks, herd movement, borderland market timing",
    confidence: "Medium",
    storySummary:
      "Abrupt rainfall shifts are pulling herds toward a smaller number of viable water points before markets and local services can adjust.",
    thresholdReason:
      "Uneven rainfall changes grazing routes and local price pressure before it appears as a visible incident spike.",
    communityContext:
      "Many communities across this belt are pastoral and Cushitic-speaking, so rainfall shifts quickly affect livestock movement and support routes.",
    watchItems: [
      "Short-horizon rainfall deviation and pasture concentration",
      "Livestock price swings and distress sales",
      "Water-point pressure and local mediation requests",
    ],
    sourceRegionId: "placeholder",
    population: "841,000",
  },
  {
    id: "north_west_frontier",
    shapeId: "north_west_frontier",
    name: "North West Frontier",
    riskLevel: "watch",
    riskScore: 0.52,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Drought onset",
    secondaryDriver: "Cross-border livestock movement",
    confidence: "Low",
    storySummary: "Emerging drought conditions are beginning to affect pastoral routes.",
    thresholdReason: "Drought onset indicators rising above seasonal norms.",
    communityContext: "Predominantly agro-pastoral communities with seasonal migration patterns.",
    watchItems: ["Rainfall accumulation deficit", "Livestock body condition scores"],
    sourceRegionId: "placeholder",
    population: "330,000",
  },
  {
    id: "lake_basin",
    shapeId: "lake_basin",
    name: "Lake Basin Region",
    riskLevel: "watch",
    riskScore: 0.47,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Flooding risk",
    secondaryDriver: "Resource competition",
    confidence: "Medium",
    storySummary: "Elevated lake levels are creating localized flooding risk.",
    thresholdReason: "Lake levels approaching seasonal flood thresholds.",
    communityContext: "Fishing and farming communities at lake margins.",
    watchItems: ["Lake water levels", "Fish market pricing"],
    sourceRegionId: "placeholder",
    population: "980,000",
  },
  {
    id: "upper_eastern_corridor",
    shapeId: "upper_eastern_corridor",
    name: "Upper Eastern Corridor",
    riskLevel: "low",
    riskScore: 0.28,
    thresholdStatus: "Below threshold",
    primaryDriver: "Seasonal drought risk",
    secondaryDriver: "Land conflict",
    confidence: "High",
    storySummary: "Currently stable with low-level monitoring.",
    thresholdReason: "Seasonal rainfall expectations met.",
    communityContext: "Mixed farming communities with established coping mechanisms.",
    watchItems: ["Crop failure indicators"],
    sourceRegionId: "placeholder",
    population: "580,000",
  },
  {
    id: "central_highlands",
    shapeId: "central_highlands",
    name: "Central Highlands",
    riskLevel: "low",
    riskScore: 0.18,
    thresholdStatus: "Below threshold",
    primaryDriver: "Political sensitivity",
    secondaryDriver: "Election period",
    confidence: "High",
    storySummary: "Stable conditions with commercial agriculture performing normally.",
    thresholdReason: "Rainfall and economic indicators within expected range.",
    communityContext: "Commercial agriculture, strong market linkages.",
    watchItems: ["Election-period tension indicators"],
    sourceRegionId: "placeholder",
    population: "1,200,000",
  },
  {
    id: "nairobi_metro",
    shapeId: "nairobi_metro",
    name: "Nairobi Metropolitan",
    riskLevel: "low",
    riskScore: 0.22,
    thresholdStatus: "Below threshold",
    primaryDriver: "Urban unrest risk",
    secondaryDriver: "Economic inequality",
    confidence: "High",
    storySummary: "Stable. Monitor urban cost-of-living triggers.",
    thresholdReason: "Urban risk signals below threshold.",
    communityContext: "High-density urban population, service economy.",
    watchItems: ["Cost of living index", "Urban unemployment rates"],
    sourceRegionId: "placeholder",
    population: "4,400,000",
  },
  {
    id: "coast_belt",
    shapeId: "coast_belt",
    name: "Coast Belt",
    riskLevel: "watch",
    riskScore: 0.44,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Radicalization corridors",
    secondaryDriver: "Economic marginalization",
    confidence: "Medium",
    storySummary: "Long-standing risk corridors under monitoring.",
    thresholdReason: "Radicalization indicators slightly elevated.",
    communityContext: "Tourism, trade, and fishing communities. Historical marginalization.",
    watchItems: ["Security incident frequency", "Youth unemployment indicators"],
    sourceRegionId: "placeholder",
    population: "1,100,000",
  },
  {
    id: "south_rift",
    shapeId: "south_rift",
    name: "South Rift Valley",
    riskLevel: "watch",
    riskScore: 0.41,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Livestock theft",
    secondaryDriver: "Land boundary disputes",
    confidence: "Medium",
    storySummary: "Seasonal tension elevation in pastoral zones.",
    thresholdReason: "Cross-community livestock theft slightly elevated.",
    communityContext: "Pastoral and small-holder communities with inter-ethnic friction points.",
    watchItems: ["Livestock theft incident rate", "Mediation requests"],
    sourceRegionId: "placeholder",
    population: "630,000",
  },
];
