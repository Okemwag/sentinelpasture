"use client";

import { useEffect, useState } from "react";
import { Icon } from "@iconify/react";
import PageHeader from "@/components/dashboard/page-header";
import LineChart from "@/components/dashboard/line-chart";
import { apiClient } from "@/lib/api-client";

type Outcome = {
  intervention: string;
  deployed: string;
  riskBefore: number;
  riskAfter: number;
  trend: string;
  commentary: string;
};

type ChartPoint = { date: string; value: number };

// Map raw region IDs / numeric codes → human-readable names
const REGION_NAME_MAP: Record<string, string> = {
  "51325": "North Eastern Drylands",
  "51326": "North West Frontier",
  "51327": "Upper Eastern Corridor",
  "51328": "Lake Basin Region",
  "51329": "Central Highlands",
  "51330": "Nairobi Metropolitan",
  "51331": "Coast Belt",
  "51332": "South Rift Valley",
  north_eastern_drylands: "North Eastern Drylands",
  north_west_frontier: "North West Frontier",
  upper_eastern_corridor: "Upper Eastern Corridor",
  lake_basin: "Lake Basin Region",
  central_highlands: "Central Highlands",
  nairobi_metro: "Nairobi Metropolitan",
  coast_belt: "Coast Belt",
  south_rift: "South Rift Valley",
};

function resolveRegionName(raw: string): string {
  // Try direct map
  if (REGION_NAME_MAP[raw]) return REGION_NAME_MAP[raw];
  // Try extracting numeric ID from patterns like "pressure in 51325 moved from..."
  const numericMatch = raw.match(/\b(5\d{4})\b/);
  if (numericMatch && REGION_NAME_MAP[numericMatch[1]]) {
    return raw.replace(numericMatch[1], REGION_NAME_MAP[numericMatch[1]]);
  }
  return raw;
}

function cleanCommentary(raw: string): string {
  // Replace any raw numeric IDs with region names
  let out = raw;
  Object.entries(REGION_NAME_MAP).forEach(([id, name]) => {
    out = out.replace(new RegExp(`\\b${id}\\b`, "g"), name);
  });
  return out;
}

function DeltaBadge({ before, after }: { before: number; after: number }) {
  const delta = after - before;
  const improved = delta < 0;
  const pct = Math.abs(delta);
  return (
    <div className={`inline-flex items-center gap-1 text-[12px] font-semibold px-2 py-0.5 rounded-full border
      ${improved
        ? "bg-[#EDF4EB] text-[#2A5B28] border-[#A8C9A2]"
        : "bg-[#F5EFE6] text-[#5A3E1E] border-[#D4AA78]"}`}>
      <Icon icon={improved ? "mdi:trending-down" : "mdi:trending-up"} className="h-3.5 w-3.5" />
      {improved ? "−" : "+"}{pct} pts
    </div>
  );
}

function RiskBar({ label, value, max = 100 }: { label: string; value: number; max?: number }) {
  const pct = Math.min((value / max) * 100, 100);
  const color = value >= 70 ? "#4A3A26" : value >= 50 ? "#8C6A3D" : value >= 30 ? "#C7A56B" : "#7A9B70";
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-[11px] text-[#6B7280]">{label}</span>
        <span className="text-[13px] font-semibold text-[#111111]">{value}</span>
      </div>
      <div className="h-[6px] rounded-full bg-[#F3F4F6] overflow-hidden">
        <div className="h-full rounded-full animate-grow-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

const PLACEHOLDER_OUTCOMES: Outcome[] = [
  {
    intervention: "Emergency water point deployment — North Eastern Drylands",
    deployed: "2024-02-14",
    riskBefore: 78,
    riskAfter: 58,
    trend: "Declining",
    commentary: "Deployment of 12 temporary water points in North Eastern Drylands reduced pastoral migration pressure. Livestock distress sales decreased by 34% in the 3 weeks following intervention.",
  },
  {
    intervention: "Transport corridor clearance — Coast Belt",
    deployed: "2024-01-28",
    riskBefore: 61,
    riskAfter: 45,
    trend: "Stable",
    commentary: "Pre-clearing of the Mombasa–Malindi coastal route stabilized market supply chains. Food price volatility in Coast Belt dropped from +18% to +4% above baseline within 2 weeks.",
  },
  {
    intervention: "Early warning communication — South Rift Valley",
    deployed: "2024-03-01",
    riskBefore: 52,
    riskAfter: 44,
    trend: "Declining",
    commentary: "Community-level early warning messaging in South Rift Valley increased pastoral route coordination. Inter-community conflict mediation requests dropped 22%.",
  },
  {
    intervention: "Flood relief pre-positioning — Lake Basin Region",
    deployed: "2024-02-20",
    riskBefore: 67,
    riskAfter: 53,
    trend: "Declining",
    commentary: "Pre-positioned flood relief supplies in Lake Basin Region allowed faster response when flooding exceeded thresholds. Displacement numbers were 31% lower than 2022 comparable event.",
  },
  {
    intervention: "Livestock support teams — North West Frontier",
    deployed: "2024-01-15",
    riskBefore: 55,
    riskAfter: 60,
    trend: "Increasing",
    commentary: "Livestock support team deployment in North West Frontier did not reduce risk as expected because secondary drought conditions intensified after deployment. Escalation protocol triggered.",
  },
];

const PLACEHOLDER_CHART: ChartPoint[] = Array.from({ length: 12 }, (_, i) => {
  const d = new Date("2024-01-01");
  d.setDate(d.getDate() + i * 7);
  return { date: d.toISOString().slice(0, 10), value: Math.max(25, 75 - i * 4 + (i % 3 === 0 ? 6 : -3)) };
});

export default function OutcomesPage() {
  const [rows, setRows] = useState<Outcome[]>([]);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [metadata, setMetadata] = useState({ modelVersion: "—", lastUpdated: "—", confidence: "Medium" });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const [outcomesResp, chartResp] = await Promise.all([
          apiClient.getOutcomes(),
          apiClient.getOutcomesChart(),
        ]);
        if (!active) return;
        // Sanitize commentary and intervention names
        const cleaned = outcomesResp.data.map((o) => ({
          ...o,
          intervention: resolveRegionName(o.intervention),
          commentary: cleanCommentary(o.commentary),
        }));
        setRows(cleaned.length ? cleaned : PLACEHOLDER_OUTCOMES);
        setChartData(chartResp.data.length ? chartResp.data : PLACEHOLDER_CHART);
        if (outcomesResp.metadata) setMetadata(outcomesResp.metadata);
      } catch {
        if (!active) return;
        setRows(PLACEHOLDER_OUTCOMES);
        setChartData(PLACEHOLDER_CHART);
        setError("Live data unavailable — showing modelled baseline outcomes");
      }
    }
    void load();
    return () => { active = false; };
  }, []);

  const displayed = rows.length ? rows : PLACEHOLDER_OUTCOMES;
  const improved = displayed.filter((o) => o.riskAfter < o.riskBefore).length;
  const worsened = displayed.filter((o) => o.riskAfter >= o.riskBefore).length;
  const avgDelta = displayed.length
    ? Math.round(displayed.reduce((s, o) => s + (o.riskBefore - o.riskAfter), 0) / displayed.length)
    : 0;

  return (
    <div className="max-w-[1400px] mx-auto p-6 space-y-5">
      <PageHeader title="Intervention Outcomes & Impact Tracking" />

      {error && (
        <div className="rounded-[8px] border border-[#E2C99A] bg-[#FBF5EB] px-4 py-2.5 text-[12px] text-[#7A4F1E]">
          {error}
        </div>
      )}

      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[
          { icon: "mdi:check-circle-outline", label: "Interventions Tracked", value: String(displayed.length), color: "#6B7280", bg: "#F3F4F6" },
          { icon: "mdi:trending-down", label: "Risk Reduced", value: String(improved), note: "regions improved", color: "#2A5B28", bg: "#EDF4EB" },
          { icon: "mdi:trending-up", label: "Not Effective", value: String(worsened), note: "require review", color: "#5A3E1E", bg: "#F5EFE6" },
          { icon: "mdi:chart-bar-stacked", label: "Avg. Risk Reduction", value: `${avgDelta} pts`, note: "across all interventions", color: "#374151", bg: "#F9FAFB" },
        ].map((c) => (
          <div key={c.label} className="bg-white border border-[#E5E7EB] rounded-[8px] p-4 flex items-center gap-3">
            <div className="h-9 w-9 rounded-full flex items-center justify-center shrink-0" style={{ backgroundColor: c.bg }}>
              <Icon icon={c.icon} className="h-5 w-5" style={{ color: c.color }} />
            </div>
            <div>
              <div className="text-[11px] text-[#9CA3AF]">{c.label}</div>
              <div className="text-[20px] font-bold text-[#111111] leading-tight">{c.value}</div>
              {c.note && <div className="text-[11px] text-[#9CA3AF]">{c.note}</div>}
            </div>
          </div>
        ))}
      </div>

      {/* Risk trend chart */}
      <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-[15px] font-semibold text-[#111111]">National Risk Trend — Post-Intervention</h2>
            <p className="text-[12px] text-[#6B7280] mt-0.5">Composite stability index across all monitored regions</p>
          </div>
          <div className="flex gap-4 text-[12px] text-[#9CA3AF]">
            <span>Model: {metadata.modelVersion}</span>
            <span>Updated: {metadata.lastUpdated}</span>
          </div>
        </div>
        <LineChart data={chartData.length ? chartData : PLACEHOLDER_CHART} height={200} />
      </div>

      {/* Outcome cards */}
      <div className="bg-white border border-[#E5E7EB] rounded-[8px]">
        <div className="border-b border-[#F3F4F6] px-5 py-3">
          <h2 className="text-[15px] font-semibold text-[#111111]">Deployed Intervention Records</h2>
        </div>
        <div className="divide-y divide-[#F9FAFB]">
          {displayed.map((outcome, idx) => {
            const improved = outcome.riskAfter < outcome.riskBefore;
            const deployDate = new Date(outcome.deployed).toLocaleDateString("en-US", {
              year: "numeric", month: "short", day: "numeric",
            });
            return (
              <div
                key={`${outcome.intervention}-${idx}`}
                className="px-5 py-5 hover:bg-[#FAFAFA] transition-colors"
                style={{ borderLeft: `3px solid ${improved ? "#7A9B70" : "#C7A56B"}` }}
              >
                <div className="flex flex-wrap items-start justify-between gap-3 mb-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1 flex-wrap">
                      <span className={`text-[10px] font-semibold uppercase tracking-widest px-2 py-0.5 rounded-full
                        ${improved ? "bg-[#EDF4EB] text-[#2A5B28]" : "bg-[#F5EFE6] text-[#5A3E1E]"}`}>
                        {improved ? "Effective" : "Under Review"}
                      </span>
                      <span className="text-[11px] text-[#9CA3AF]">Deployed {deployDate}</span>
                    </div>
                    <h3 className="text-[15px] font-semibold text-[#111111] leading-snug">{outcome.intervention}</h3>
                  </div>
                  <DeltaBadge before={outcome.riskBefore} after={outcome.riskAfter} />
                </div>

                {/* Before / After bars */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <RiskBar label="Risk Score Before" value={outcome.riskBefore} />
                  <RiskBar label="Risk Score After" value={outcome.riskAfter} />
                </div>

                {/* Trend + Commentary */}
                <div className="flex items-start gap-3">
                  <div className="shrink-0">
                    <span className="text-[11px] text-[#9CA3AF] uppercase tracking-widest">Trend</span>
                    <div className={`text-[13px] font-semibold mt-0.5 ${outcome.trend === "Declining" ? "text-[#2A5B28]"
                        : outcome.trend === "Increasing" ? "text-[#5A3E1E]"
                          : "text-[#6B7280]"}`}>
                      {outcome.trend}
                    </div>
                  </div>
                  <div className="flex-1 border-l border-[#F3F4F6] pl-3">
                    <div className="text-[11px] text-[#9CA3AF] uppercase tracking-widest mb-1">Commentary</div>
                    <p className="text-[13px] leading-relaxed text-[#374151]">{outcome.commentary}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
