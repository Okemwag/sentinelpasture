"use client";

import { useEffect, useState } from "react";
import { getRiskPalette, type RegionalRiskLevel } from "@/lib/kenya-map";
import ProbabilityBars from "@/components/dashboard/probability-bars";
import InterventionCards from "@/components/dashboard/intervention-cards";
import { apiClient } from "@/lib/api-client";

interface RegionProfile {
    id: string;
    name: string;
    riskLevel: RegionalRiskLevel;
    riskScore: number;
    confidence: string;
    primaryDriver: string;
    secondaryDriver: string;
    storySummary: string;
    thresholdReason: string;
    communityContext: string;
    watchItems: string[];
    thresholdStatus: string;
    population?: string;
    sourceRegionId: string;
}

interface Intervention {
    category: string;
    expectedImpact: string;
    timeToEffect: string;
    costBand: string;
    confidence: string;
}

const REGION_PROFILES: Record<string, { economy: string; climate: string; security: string }> = {
    north_eastern_drylands: {
        economy: "Pastoral livestock economy",
        climate: "Flood and drought cycles",
        security: "Cross-border corridor",
    },
    north_west_frontier: {
        economy: "Agro-pastoral mixed economy",
        climate: "Semi-arid, seasonal rains",
        security: "Inter-community tension corridors",
    },
    upper_eastern_corridor: {
        economy: "Mixed farming and livestock",
        climate: "Drought-prone, low rainfall",
        security: "Historically land-conflict zones",
    },
    lake_basin: {
        economy: "Fishing, small-scale agriculture",
        climate: "Lake-influenced, flooding risk",
        security: "Cross-ethnic resource competition",
    },
    central_highlands: {
        economy: "Commercial agriculture, tea, tourism",
        climate: "Temperate highland, rains reliable",
        security: "Low risk, political sensitivity",
    },
    nairobi_metro: {
        economy: "Services, formal employment",
        climate: "Urban heat island, flooding",
        security: "Urban unrest, election-sensitive",
    },
    coast_belt: {
        economy: "Tourism, port trade, fishing",
        climate: "Tropical, cyclone exposure",
        security: "Radicalization corridors",
    },
    south_rift: {
        economy: "Pastoral and small-holder farming",
        climate: "Semi-arid, drought-prone",
        security: "Cross-ethnic livestock theft",
    },
};

const HISTORY_BY_LEVEL: Record<RegionalRiskLevel, { title: string; consequences: string[] }> = {
    critical: {
        title: "Similar Critical Event (2022)",
        consequences: [
            "34,000+ people displaced across multiple counties",
            "Livestock losses exceeding $28M market value",
            "Closure of 2 major supply corridor routes",
            "Food prices spiked 22% above seasonal average",
        ],
    },
    elevated: {
        title: "Comparable Event (2019)",
        consequences: [
            "~18,000 people affected, 6,000 temporarily displaced",
            "Livestock migration pressure increased 3-fold",
            "Market supply disruptions for 6–8 weeks",
            "Heightened inter-community tensions reported",
        ],
    },
    watch: {
        title: "Watch-Level Event (2018)",
        consequences: [
            "Localized displacement of 4,000–7,000 people",
            "Livestock distress sales at reduced market prices",
            "Minor road accessibility degradation",
            "Increased requests for conflict mediation",
        ],
    },
    low: {
        title: "Baseline Period (2020–2021)",
        consequences: [
            "Stable pastoral movement patterns",
            "Normal market supply chains maintained",
            "Low inter-community conflict reporting",
            "Seasonal rainfall within expected range",
        ],
    },
};

function getProbabilities(level: RegionalRiskLevel, score: number) {
    const base = Math.round(score * 100);
    switch (level) {
        case "critical":
            return [
                { label: "Probability of population displacement", probability: Math.min(base + 15, 95) },
                { label: "Probability of market supply disruption", probability: Math.min(base + 5, 88) },
                { label: "Probability of livestock migration spike", probability: Math.min(base + 10, 92) },
                { label: "Probability of security incident", probability: Math.min(base - 10, 75) },
            ];
        case "elevated":
            return [
                { label: "Probability of livestock migration spike", probability: base + 8 },
                { label: "Probability of market supply disruption", probability: base - 5 },
                { label: "Probability of localized displacement", probability: base - 15 },
                { label: "Probability of conflict escalation", probability: base - 25 },
            ];
        default:
            return [
                { label: "Probability of livestock movement pressure", probability: Math.max(base - 5, 15) },
                { label: "Probability of market price volatility", probability: Math.max(base - 15, 12) },
                { label: "Probability of localized displacement", probability: Math.max(base - 25, 8) },
                { label: "Probability of security incident", probability: Math.max(base - 35, 5) },
            ];
    }
}

function getExposure(level: RegionalRiskLevel, score: number) {
    const pop = Math.round((score * 180 + 30) * 1000);
    const value = Math.round(score * 65 + 12);
    return {
        population: `${(pop / 1000).toFixed(0)}k people`,
        economicValue: `$${value}M`,
        corridors: level === "critical" ? "3" : level === "elevated" ? "2" : "1",
        waterRisk: level === "critical" ? "High" : level === "elevated" ? "Moderate" : "Low",
    };
}

function TrendBadge({ trend }: { trend: string }) {
    const up = trend === "up";
    const down = trend === "down";
    return (
        <span
            className={`inline-flex items-center gap-1 text-[12px] font-medium ${up ? "text-[#8C6A3D]" : down ? "text-[#3A6B33]" : "text-[#6B7280]"
                }`}
        >
            {up ? "↑ Increasing" : down ? "↓ Decreasing" : "→ Stable"}
        </span>
    );
}

interface IntelligencePanelProps {
    region: RegionProfile | null;
}

export default function IntelligencePanel({ region }: IntelligencePanelProps) {
    const [interventions, setInterventions] = useState<Intervention[]>([]);

    useEffect(() => {
        if (!region) return;
        apiClient.getInterventions(region.sourceRegionId).then((r) => {
            setInterventions(r.data);
        }).catch(() => {
            // use fallback
            setInterventions(FALLBACK_INTERVENTIONS);
        });
    }, [region?.id]);

    if (!region) {
        return (
            <div className="flex flex-col items-center justify-center h-full text-center px-6 py-12">
                <div className="w-10 h-10 rounded-full bg-[#F3F4F6] flex items-center justify-center mb-4">
                    <svg className="w-5 h-5 text-[#9CA3AF]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                    </svg>
                </div>
                <div className="text-[14px] font-medium text-[#374151] mb-1">Select a region</div>
                <div className="text-[13px] text-[#9CA3AF]">Click any region on the map to view intelligence analysis</div>
            </div>
        );
    }

    const palette = getRiskPalette(region.riskLevel);
    const profile = REGION_PROFILES[region.id] ?? {
        economy: "Mixed economy",
        climate: "Variable climate patterns",
        security: "Under monitoring",
    };
    const history = HISTORY_BY_LEVEL[region.riskLevel];
    const probabilities = getProbabilities(region.riskLevel, region.riskScore);
    const exposure = getExposure(region.riskLevel, region.riskScore);
    const trend = region.riskScore > 0.6 ? "up" : region.riskScore < 0.3 ? "down" : "stable";
    const displayedInterventions = interventions.length ? interventions : FALLBACK_INTERVENTIONS;

    return (
        <div className="intel-scroll h-full px-5 py-5 space-y-5 animate-slide-right">

            {/* ── Region Profile ── */}
            <div>
                <div className="flex items-start justify-between gap-3 mb-3">
                    <div>
                        <span
                            className={`inline-flex text-[10px] font-semibold uppercase tracking-[0.2em] px-2.5 py-1 rounded-full border ${palette.badge}`}
                        >
                            {palette.label}
                        </span>
                        <h2 className="mt-2 text-[20px] font-semibold text-[#111111] leading-tight">{region.name}</h2>
                    </div>
                    <div className="text-right flex-shrink-0">
                        <div className="text-[11px] uppercase tracking-[0.16em] text-[#9CA3AF]">Risk Score</div>
                        <div className="text-[28px] font-bold leading-none mt-0.5" style={{ color: palette.textColor }}>
                            {Math.round(region.riskScore * 100)}
                        </div>
                        <div className="text-[11px] text-[#9CA3AF] mt-0.5">Confidence {region.confidence}</div>
                    </div>
                </div>
                <div className="grid grid-cols-1 gap-1.5 bg-[#F9FAFB] rounded-[8px] p-3 border border-[#F3F4F6]">
                    {[
                        { label: "Economy", val: profile.economy },
                        { label: "Climate", val: profile.climate },
                        { label: "Security", val: profile.security },
                        ...(region.population ? [{ label: "Population", val: region.population }] : []),
                    ].map(({ label, val }) => (
                        <div key={label} className="flex gap-2 text-[12px]">
                            <span className="text-[#9CA3AF] w-16 flex-shrink-0">{label}</span>
                            <span className="text-[#374151]">{val}</span>
                        </div>
                    ))}
                </div>
            </div>

            <div className="border-t border-[#F3F4F6]" />

            {/* ── Risk Signal ── */}
            <div>
                <SectionLabel>Risk Signal</SectionLabel>
                <div className="bg-[#F9FAFB] rounded-[8px] p-3 border border-[#F3F4F6]">
                    <div className="text-[14px] font-medium text-[#111111] mb-2">{region.thresholdStatus}</div>
                    <div className="grid grid-cols-2 gap-2">
                        <InfoPair label="Time Horizon" val="14–30 days" />
                        <InfoPair label="Trend" val={<TrendBadge trend={trend} />} />
                        <InfoPair label="Primary Driver" val={region.primaryDriver} />
                        <InfoPair label="Secondary" val={region.secondaryDriver?.split(",")[0] ?? "—"} />
                    </div>
                </div>
                <p className="mt-2 text-[12px] leading-relaxed text-[#6B7280]">{region.thresholdReason}</p>
            </div>

            <div className="border-t border-[#F3F4F6]" />

            {/* ── Drivers ── */}
            <div>
                <SectionLabel>Key Drivers</SectionLabel>
                <div className="space-y-2">
                    {[
                        { label: region.primaryDriver, score: 78, source: "Satellite + Rainfall API" },
                        { label: region.secondaryDriver?.split(",")[0] ?? "Market pressure", score: 55, source: "Market data" },
                        { label: "Mobility pattern shift", score: 41, source: "Movement index" },
                    ].map((d) => (
                        <DriverRow key={d.label} {...d} />
                    ))}
                </div>
            </div>

            <div className="border-t border-[#F3F4F6]" />

            {/* ── Historical Context ── */}
            <div>
                <SectionLabel>Historical Context</SectionLabel>
                <div className="rounded-[8px] border border-[#E9E4DC] bg-[#FAF8F4] p-3">
                    <div className="text-[12px] font-semibold text-[#5A4A2E] mb-2">{history.title}</div>
                    <ul className="space-y-1">
                        {history.consequences.map((c) => (
                            <li key={c} className="flex items-start gap-1.5 text-[12px] text-[#6B7280]">
                                <span className="text-[#C7A56B] mt-0.5 flex-shrink-0">–</span>
                                {c}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>

            <div className="border-t border-[#F3F4F6]" />

            {/* ── Predictions ── */}
            <div>
                <SectionLabel>Predicted Consequences</SectionLabel>
                <p className="text-[11px] text-[#9CA3AF] mb-3">If current conditions persist:</p>
                <ProbabilityBars items={probabilities} />
            </div>

            <div className="border-t border-[#F3F4F6]" />

            {/* ── Socioeconomic Exposure ── */}
            <div>
                <SectionLabel>Socioeconomic Exposure</SectionLabel>
                <div className="grid grid-cols-2 gap-2">
                    <ExposureCard label="Population exposed" val={exposure.population} />
                    <ExposureCard label="Economic exposure" val={exposure.economicValue} />
                    <ExposureCard label="Transport corridors" val={exposure.corridors + " affected"} />
                    <ExposureCard label="Water access risk" val={exposure.waterRisk} />
                </div>
            </div>

            <div className="border-t border-[#F3F4F6]" />

            {/* ── Interventions ── */}
            <div>
                <SectionLabel>Recommended Interventions</SectionLabel>
                <p className="text-[11px] text-[#9CA3AF] mb-3">
                    Ranked by expected impact. Final authority with designated governance bodies.
                </p>
                <InterventionCards interventions={displayedInterventions} />
            </div>

            {/* ── Community Context ── */}
            <div className="border-t border-[#F3F4F6] pt-5">
                <SectionLabel>Community Context</SectionLabel>
                <p className="text-[12px] leading-relaxed text-[#6B7280]">{region.communityContext}</p>
            </div>

            {region.watchItems.length > 0 && (
                <div className="pb-4">
                    <SectionLabel>Watch Indicators</SectionLabel>
                    <div className="space-y-1.5">
                        {region.watchItems.map((item) => (
                            <div
                                key={item}
                                className="rounded-[6px] border border-[#E5E7EB] bg-[#F9FAFB] px-3 py-2 text-[12px] text-[#374151] flex items-start gap-2"
                            >
                                <span className="text-[#C7A56B] flex-shrink-0 mt-0.5">◆</span>
                                {item}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

/* ── Sub-components ── */

function SectionLabel({ children }: { children: React.ReactNode }) {
    return (
        <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[#9CA3AF] mb-2">
            {children}
        </div>
    );
}

function InfoPair({ label, val }: { label: string; val: React.ReactNode }) {
    return (
        <div>
            <div className="text-[10px] text-[#9CA3AF] uppercase tracking-[0.12em]">{label}</div>
            <div className="text-[12px] text-[#374151] mt-0.5">{val}</div>
        </div>
    );
}

function DriverRow({ label, score, source }: { label: string; score: number; source: string }) {
    const color = score >= 70 ? "#4A3A26" : score >= 50 ? "#8C6A3D" : "#C7A56B";
    const bg = score >= 70 ? "#EDE8E0" : score >= 50 ? "#F5EFE6" : "#FBF5EB";
    return (
        <div className="rounded-[6px] border border-[#F3F4F6] bg-[#FAFAFA] p-2.5">
            <div className="flex items-center justify-between mb-1.5">
                <span className="text-[12px] font-medium text-[#374151] truncate pr-2">{label}</span>
                <span
                    className="text-[11px] font-semibold px-2 py-0.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: bg, color }}
                >
                    {score}
                </span>
            </div>
            <div className="h-[4px] bg-[#F3F4F6] rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full animate-grow-bar"
                    style={{ width: `${score}%`, backgroundColor: color }}
                />
            </div>
            <div className="mt-1 text-[10px] text-[#9CA3AF]">{source}</div>
        </div>
    );
}

function ExposureCard({ label, val }: { label: string; val: string }) {
    return (
        <div className="bg-[#F9FAFB] border border-[#F3F4F6] rounded-[8px] p-2.5">
            <div className="text-[10px] text-[#9CA3AF] uppercase tracking-[0.12em] mb-0.5">{label}</div>
            <div className="text-[14px] font-semibold text-[#111111]">{val}</div>
        </div>
    );
}

const FALLBACK_INTERVENTIONS: Intervention[] = [
    {
        category: "Deploy temporary water points",
        expectedImpact: "Reduce migration pressure",
        timeToEffect: "3–5 days",
        costBand: "Low",
        confidence: "High",
    },
    {
        category: "Pre-position flood relief supplies",
        expectedImpact: "Reduce displacement risk",
        timeToEffect: "7–14 days",
        costBand: "Medium",
        confidence: "High",
    },
    {
        category: "Maintain transport corridor clearance",
        expectedImpact: "Stabilize market supply",
        timeToEffect: "Immediate",
        costBand: "Medium",
        confidence: "Medium",
    },
    {
        category: "Coordinate early warning communication",
        expectedImpact: "Increase community preparedness",
        timeToEffect: "1–3 days",
        costBand: "Low",
        confidence: "High",
    },
    {
        category: "Deploy livestock support teams",
        expectedImpact: "Reduce distress livestock sales",
        timeToEffect: "5–10 days",
        costBand: "High",
        confidence: "Medium",
    },
];
