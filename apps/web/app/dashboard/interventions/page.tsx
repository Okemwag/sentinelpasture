"use client";

import { useEffect, useState } from "react";
import PageHeader from "@/components/dashboard/page-header";
import InterventionCards from "@/components/dashboard/intervention-cards";
import { apiClient } from "@/lib/api-client";

type Intervention = {
  category: string;
  expectedImpact: string;
  timeToEffect: string;
  costBand: string;
  confidence: string;
};

const COST_DEFINITIONS: Record<string, string> = {
  Low: "Under KES 5M — can be mobilised rapidly from existing budgets",
  Medium: "KES 5M–50M — requires inter-agency coordination",
  High: "Over KES 50M — requires cabinet-level approval",
};

export default function InterventionsPage() {
  const [rows, setRows] = useState<Intervention[]>([]);
  const [metadata, setMetadata] = useState<{
    modelVersion: string;
    lastUpdated: string;
    confidence: string;
    regionName?: string;
    regionId?: string;
    riskScore?: number;
    riskLevel?: string;
    thresholdStatus?: string;
    primaryDriver?: string;
    thresholdReason?: string;
  }>({ modelVersion: "—", lastUpdated: "—", confidence: "Medium" });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    apiClient.getInterventions().then((r) => {
      if (!active) return;
      setRows(r.data);
      if (r.metadata) setMetadata(r.metadata);
    }).catch((err) => {
      if (!active) return;
      setError(err instanceof Error ? err.message : "Unable to load interventions");
    });
    return () => { active = false; };
  }, []);

  const hasData = rows.length > 0;
  const lowCost = rows.filter((r) => r.costBand === "Low").length;
  const mediumCost = rows.filter((r) => r.costBand === "Medium").length;
  const highCost = rows.filter((r) => r.costBand === "High").length;
  const highConfidence = rows.filter((r) => r.confidence === "High").length;

  return (
    <div className="max-w-[1400px] mx-auto p-6 space-y-6">
      <PageHeader title="Intervention Analysis" />

      {error && (
        <div className="rounded-[8px] border border-[#E2C99A] bg-[#FBF5EB] p-3 text-[13px] text-[#7A4F1E]">
          {error}
        </div>
      )}

      {/* Policy note + target region context */}
      <div className="rounded-[8px] border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 flex flex-col gap-2">
        <div className="flex items-start gap-3">
          <svg className="h-4 w-4 text-[#9CA3AF] shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <p className="text-[12px] leading-relaxed text-[#6B7280]">
            Recommendations are generated from the highest current modelled pressure zone unless a region filter is supplied.
            Final decision authority remains with designated governance bodies.
          </p>
        </div>
        {metadata.regionName && (
          <div className="mt-1 rounded-[6px] bg-white border border-[#E5E7EB] px-3 py-2 flex flex-wrap items-start gap-2 text-[12px] text-[#4B5563]">
            <span className="font-semibold text-[#111827]">
              Target region: {metadata.regionName}
            </span>
            {typeof metadata.riskScore === "number" && (
              <span className="text-[#6B7280]">
                · Risk score {Math.round(Number(metadata.riskScore) * 100)}
              </span>
            )}
            {metadata.thresholdStatus && (
              <span className="text-[#6B7280]">
                · {metadata.thresholdStatus}
              </span>
            )}
            {metadata.primaryDriver && (
              <span className="text-[#6B7280]">
                · Primary driver: {metadata.primaryDriver}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Metadata + quick takeaway */}
      <div className="space-y-2">
        <div className="flex flex-wrap gap-4 text-[12px] text-[#9CA3AF]">
          <span>Model: {metadata.modelVersion}</span>
          <span>Updated: {metadata.lastUpdated}</span>
          <span>Confidence: {metadata.confidence}</span>
        </div>
        {hasData && (
          <div className="text-[13px] text-[#4B5563] bg-[#F9FAFB] border border-[#E5E7EB] rounded-[8px] px-4 py-2">
            <span className="font-semibold text-[#111827]">
              {rows.length} ranked options
            </span>
            <span>
              {" "}with a focus on{" "}
              <strong>{lowCost} low</strong>,{" "}
              <strong>{mediumCost} medium</strong>, and{" "}
              <strong>{highCost} high</strong> cost-band interventions;{" "}
              <strong>{highConfidence}</strong> are high-confidence recommendations.
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_300px]">
        {/* Main card list */}
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
          <div className="mb-3">
            <h2 className="text-[15px] font-semibold text-[#111111]">
              Ranked Intervention Options
            </h2>
            {metadata.regionName && (
              <p className="mt-1 text-[12px] text-[#6B7280]">
                For <strong>{metadata.regionName}</strong>
                {metadata.thresholdStatus && <> — {metadata.thresholdStatus}</>}
                {metadata.primaryDriver && <> · primary driver: {metadata.primaryDriver}</>}
              </p>
            )}
          </div>
          {rows.length === 0 ? (
            <div className="text-[13px] text-[#9CA3AF]">
              No ranked intervention options are currently available from the model.
            </div>
          ) : (
            <InterventionCards
              interventions={rows}
              context={{
                regionName: metadata.regionName,
                thresholdStatus: metadata.thresholdStatus,
                primaryDriver: metadata.primaryDriver,
                thresholdReason: metadata.thresholdReason,
              }}
            />
          )}
        </div>

        {/* Side: cost band definitions */}
        <div className="space-y-4">
          <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
            <h3 className="text-[13px] font-semibold text-[#111111] mb-4 uppercase tracking-widest">Cost Band Guide</h3>
            <div className="space-y-3">
              {Object.entries(COST_DEFINITIONS).map(([band, desc]) => {
                const colors: Record<string, { bg: string; text: string; border: string }> = {
                  Low: { bg: "#EDF4EB", text: "#3A6B33", border: "#A8C9A2" },
                  Medium: { bg: "#FBF5EB", text: "#7A4F1E", border: "#E2C99A" },
                  High: { bg: "#EDE8E0", text: "#2A1E12", border: "#BFB099" },
                };
                const c = colors[band];
                return (
                  <div key={band}>
                    <span
                      className="inline-block text-[11px] font-semibold px-2 py-0.5 rounded-full border mb-1"
                      style={{ backgroundColor: c.bg, color: c.text, borderColor: c.border }}
                    >
                      {band}
                    </span>
                    <p className="text-[12px] text-[#6B7280] leading-relaxed">{desc}</p>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
            <h3 className="text-[13px] font-semibold text-[#111111] mb-3 uppercase tracking-widest">Confidence Levels</h3>
            <div className="space-y-2 text-[12px] text-[#6B7280]">
              <div><strong className="text-[#3A6B33]">High</strong> — Based on multiple validated data sources</div>
              <div><strong className="text-[#8C6A3D]">Medium</strong> — Modelled with partial validation</div>
              <div><strong className="text-[#9CA3AF]">Low</strong> — Inferred from incomplete signals</div>
            </div>
            {hasData && (
              <p className="mt-3 text-[12px] text-[#6B7280]">
                Interventions at the top of the list are ranked against the current highest-pressure region,
                using the same climate, market, mobility, and incident drivers shown on the Drivers page.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
// No local placeholder interventions; UI shows explicit empty states when none are available.
