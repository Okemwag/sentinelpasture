"use client";

import { useEffect, useState } from "react";

import { apiClient } from "@/lib/api-client";
import { KENYA_OUTLINE_PATH, KENYA_REGION_SHAPES, getRiskPalette, type RegionalRiskLevel } from "@/lib/kenya-map";

type MapRegion = {
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
};

export default function RiskMap() {
  const [regions, setRegions] = useState<MapRegion[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<MapRegion | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await apiClient.getRegionalMap();
        if (!active) {
          return;
        }
        const sorted = [...response.data].sort((left, right) => right.riskScore - left.riskScore);
        setRegions(sorted);
        setSelectedRegion(sorted[0] || null);
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
  const activeRegion = selectedRegion || visibleRegions[0];

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.15fr)_minmax(280px,0.85fr)]">
      {error ? (
        <div className="lg:col-span-2 mb-1 text-[13px] text-[#991B1B]">{error}</div>
      ) : null}
      <div className="rounded-[18px] border border-[#D9E3DB] bg-[linear-gradient(180deg,#F7FAF8_0%,#EDF3EF_100%)] p-4 sm:p-5">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[11px] uppercase tracking-[0.22em] text-[#5D6B63]">
              Kenya alert surface
            </div>
            <div className="mt-1 text-[15px] text-[#31413B]">
              Colors show where alert thresholds have been crossed or are close to crossing.
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-[11px]">
            {([
              ["low", "Below threshold"],
              ["watch", "Watch threshold"],
              ["elevated", "Threshold crossed"],
              ["critical", "Critical escalation"],
            ] as const).map(([level, label]) => {
              const palette = getRiskPalette(level);
              return (
                <div
                  key={level}
                  className="inline-flex items-center gap-2 rounded-full border border-[#D6DDD8] bg-white/85 px-3 py-1.5 text-[#31413B]"
                >
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: palette.fill, boxShadow: `0 0 0 1px ${palette.stroke}` }}
                  />
                  {label}
                </div>
              );
            })}
          </div>
        </div>

        <svg viewBox="0 0 420 560" className="h-[460px] w-full">
          <path
            d={KENYA_OUTLINE_PATH}
            fill="#FDFEFE"
            stroke="#9FB1A7"
            strokeWidth="5"
            strokeLinejoin="round"
          />
          {visibleRegions.map((region) => {
            const shape = KENYA_REGION_SHAPES[region.shapeId];
            if (!shape) {
              return null;
            }
            const palette = getRiskPalette(region.riskLevel);
            const isActive = activeRegion?.id === region.id;
            return (
              <g
                key={region.id}
                className="cursor-pointer transition-all duration-200"
                onMouseEnter={() => setSelectedRegion(region)}
                onClick={() => setSelectedRegion(region)}
              >
                <polygon
                  points={shape.points}
                  fill={palette.fill}
                  stroke={isActive ? "#18241F" : palette.stroke}
                  strokeWidth={isActive ? 4 : 2.2}
                  opacity={isActive ? 1 : 0.9}
                />
                <text
                  x={shape.labelX}
                  y={shape.labelY}
                  textAnchor="middle"
                  className="pointer-events-none fill-[#13211B] text-[11px] font-semibold tracking-[0.2em]"
                >
                  {shape.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {activeRegion ? (
        <div className="rounded-[18px] border border-[#E1E6E1] bg-white p-5 shadow-[0_12px_30px_rgba(28,39,34,0.08)]">
          <div className="flex items-start justify-between gap-4">
            <div>
              <span className={`inline-flex rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${getRiskPalette(activeRegion.riskLevel).badge}`}>
                {activeRegion.thresholdStatus}
              </span>
              <h3 className="mt-3 text-[22px] font-semibold text-[#111111]">{activeRegion.name}</h3>
              <p className="mt-1 text-[13px] text-[#5C6B62]">
                Source region {activeRegion.sourceRegionId}. Driver {activeRegion.primaryDriver}.
              </p>
            </div>
            <div className="rounded-[14px] bg-[#F4F7F4] px-4 py-3 text-right">
              <div className="text-[11px] uppercase tracking-[0.18em] text-[#617067]">Risk score</div>
              <div className="mt-1 text-[28px] font-semibold text-[#18241F]">
                {Math.round(activeRegion.riskScore * 100)}
              </div>
              <div className="text-[12px] text-[#617067]">Confidence {activeRegion.confidence}</div>
            </div>
          </div>

          <NarrativeBlock label="Why this threshold matters" value={activeRegion.thresholdReason} />
          <NarrativeBlock label="Operating story" value={activeRegion.storySummary} />
          <NarrativeBlock label="Community context" value={activeRegion.communityContext} />

          <div className="mt-5">
            <div className="text-[12px] font-semibold uppercase tracking-[0.18em] text-[#617067]">
              Watch next
            </div>
            <div className="mt-3 space-y-2">
              {activeRegion.watchItems.map((item) => (
                <div
                  key={item}
                  className="rounded-[12px] border border-[#E6ECE7] bg-[#F7FAF8] px-3 py-2 text-[13px] text-[#31413B]"
                >
                  {item}
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function NarrativeBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="mt-5">
      <div className="text-[12px] font-semibold uppercase tracking-[0.18em] text-[#617067]">{label}</div>
      <p className="mt-2 text-[14px] leading-6 text-[#2A3832]">{value}</p>
    </div>
  );
}

const placeholderRegions: MapRegion[] = [
  {
    id: "north_eastern_drylands",
    shapeId: "north_eastern_drylands",
    name: "North Eastern Drylands",
    riskLevel: "watch",
    riskScore: 0.52,
    thresholdStatus: "Watch threshold",
    primaryDriver: "Rainfall anomaly",
    secondaryDriver: "Pasture shocks, herd movement, and borderland market timing",
    confidence: "Medium",
    storySummary:
      "Abrupt rainfall shifts in the north eastern drylands can pull herds toward a smaller number of viable water points before markets and local services adjust.",
    thresholdReason:
      "The watch threshold matters because uneven rainfall can change grazing routes and local price pressure before it appears as a visible incident spike.",
    communityContext:
      "Many communities across this belt are pastoral and Cushitic-speaking, so rainfall shifts quickly affect livestock movement and support routes.",
    watchItems: [
      "Short-horizon rainfall deviation and pasture concentration",
      "Livestock price swings and distress sales",
      "Water-point pressure and local mediation requests",
    ],
    sourceRegionId: "loading",
  },
];
