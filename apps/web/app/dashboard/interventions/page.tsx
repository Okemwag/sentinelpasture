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
  const [metadata, setMetadata] = useState({ modelVersion: "—", lastUpdated: "—", confidence: "Medium" });
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
      setRows(PLACEHOLDER_INTERVENTIONS);
    });
    return () => { active = false; };
  }, []);

  const displayed = rows.length ? rows : PLACEHOLDER_INTERVENTIONS;

  return (
    <div className="max-w-[1400px] mx-auto p-6 space-y-6">
      <PageHeader title="Intervention Analysis" />

      {error && (
        <div className="rounded-[8px] border border-[#E2C99A] bg-[#FBF5EB] p-3 text-[13px] text-[#7A4F1E]">
          {error}
        </div>
      )}

      {/* Policy note */}
      <div className="rounded-[8px] border border-[#E5E7EB] bg-[#F9FAFB] px-4 py-3 flex items-start gap-3">
        <svg className="h-4 w-4 text-[#9CA3AF] shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-[12px] leading-relaxed text-[#6B7280]">
          Recommendations are generated from the highest current modelled pressure zone unless a region filter is supplied.
          Final decision authority remains with designated governance bodies.
        </p>
      </div>

      {/* Metadata */}
      <div className="flex flex-wrap gap-4 text-[12px] text-[#9CA3AF]">
        <span>Model: {metadata.modelVersion}</span>
        <span>Updated: {metadata.lastUpdated}</span>
        <span>Confidence: {metadata.confidence}</span>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_300px]">
        {/* Main card list */}
        <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
          <h2 className="text-[15px] font-semibold text-[#111111] mb-5">Ranked Intervention Options</h2>
          <InterventionCards interventions={displayed} />
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
          </div>
        </div>
      </div>
    </div>
  );
}

const PLACEHOLDER_INTERVENTIONS: Intervention[] = [
  { category: "Deploy temporary water points", expectedImpact: "Reduce migration pressure", timeToEffect: "3–5 days", costBand: "Low", confidence: "High" },
  { category: "Pre-position flood relief supplies", expectedImpact: "Reduce displacement risk", timeToEffect: "7–14 days", costBand: "Medium", confidence: "High" },
  { category: "Maintain transport corridor clearance", expectedImpact: "Stabilize market supply", timeToEffect: "Immediate", costBand: "Medium", confidence: "Medium" },
  { category: "Coordinate early warning communication", expectedImpact: "Increase community preparedness", timeToEffect: "1–3 days", costBand: "Low", confidence: "High" },
  { category: "Deploy livestock support teams", expectedImpact: "Reduce distress livestock sales", timeToEffect: "5–10 days", costBand: "High", confidence: "Medium" },
];
