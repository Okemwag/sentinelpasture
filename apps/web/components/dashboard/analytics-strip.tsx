"use client";

import { useState, useMemo } from "react";
import LineChart from "@/components/dashboard/line-chart";
import CausalChain from "@/components/dashboard/causal-chain";

interface ChartPoint { date: string; value: number; }
interface RegionComparison { region: string; riskScore: number; riskLevel: string; }

interface AnalyticsStripProps {
    trendData: ChartPoint[];
    comparisonData: RegionComparison[];
    selectedRegionName?: string;
}

type Tab = "risk" | "rainfall" | "market" | "security" | "mobility" | "causal" | "compare";

const COMPARISON_COLORS: Record<string, string> = {
    low: "#C8D8C2",
    watch: "#C7A56B",
    elevated: "#8C6A3D",
    critical: "#4A3A26",
};

// Generates synthetic signal data similar to real shape, different patterns per signal
function buildSignalData(seed: number, length = 16): ChartPoint[] {
    const now = new Date();
    return Array.from({ length }, (_, i) => {
        const d = new Date(now);
        d.setDate(d.getDate() - (length - 1 - i) * 7);
        const base = 40 + seed * 5;
        const wave = Math.sin((i + seed) * 0.9) * 15;
        const noise = (Math.sin(i * seed * 0.3 + 1.7) * 8);
        return {
            date: d.toISOString().slice(0, 10),
            value: Math.max(8, Math.min(98, base + wave + noise)),
        };
    });
}

const SIGNAL_TABS: { id: Tab; label: string; seed: number; strokeColor: string; fillColor: string; description: string }[] = [
    { id: "risk", label: "Risk Index", seed: 0, strokeColor: "#8C6A3D", fillColor: "#C7A56B", description: "Composite national stability index" },
    { id: "rainfall", label: "Rainfall Anomaly", seed: 1, strokeColor: "#4A7490", fillColor: "#A8C9DE", description: "% deviation from seasonal rainfall norm" },
    { id: "market", label: "Market Prices", seed: 2, strokeColor: "#7A5C96", fillColor: "#C3ADDA", description: "Food commodity price volatility index" },
    { id: "security", label: "Security Incidents", seed: 3, strokeColor: "#4A3A26", fillColor: "#B8A080", description: "Cross-border and inter-community incidents" },
    { id: "mobility", label: "Livestock Mobility", seed: 4, strokeColor: "#2E6B5A", fillColor: "#7AB8AA", description: "Pastoral migration pressure index" },
];

export default function AnalyticsStrip({ trendData, comparisonData, selectedRegionName }: AnalyticsStripProps) {
    const [tab, setTab] = useState<Tab>("risk");

    const signalCharts = useMemo(() => {
        const map: Record<string, ChartPoint[]> = {};
        SIGNAL_TABS.forEach((s) => {
            map[s.id] = s.id === "risk" && trendData.length > 0 ? trendData : buildSignalData(s.seed);
        });
        return map;
    }, [trendData]);

    const activeSignal = SIGNAL_TABS.find((s) => s.id === tab);

    const analyticsCount = comparisonData.length;

    return (
        <div className="bg-[var(--intel-s0)] border border-[var(--intel-border)] rounded-[8px] overflow-hidden">
            {/* Tab bar */}
            <div className="flex items-center border-b border-[var(--intel-border-subtle)] px-2 overflow-x-auto gap-0 no-scrollbar bg-[var(--intel-s1)]">
                {SIGNAL_TABS.map((t) => (
                    <button
                        key={t.id}
                        type="button"
                        onClick={() => setTab(t.id)}
                        className={`px-3.5 py-3 text-[12px] font-medium border-b-2 whitespace-nowrap transition-colors ${tab === t.id
                                ? "border-[var(--intel-text-primary)] text-[var(--intel-text-primary)]"
                                : "border-transparent text-[var(--intel-text-muted)] hover:text-[var(--intel-text-secondary)]"
                            }`}
                    >
                        {t.label}
                    </button>
                ))}
                <div className="mx-1 h-4 w-px bg-[var(--intel-border-subtle)] shrink-0" />
                <button
                    type="button"
                    onClick={() => setTab("causal")}
                        className={`px-3.5 py-3 text-[12px] font-medium border-b-2 whitespace-nowrap transition-colors ${tab === "causal"
                            ? "border-[var(--intel-text-primary)] text-[var(--intel-text-primary)]"
                            : "border-transparent text-[var(--intel-text-muted)] hover:text-[var(--intel-text-secondary)]"
                        }`}
                >
                    Causal Chain
                </button>
                <button
                    type="button"
                    onClick={() => setTab("compare")}
                        className={`px-3.5 py-3 text-[12px] font-medium border-b-2 whitespace-nowrap transition-colors ${tab === "compare"
                            ? "border-[var(--intel-text-primary)] text-[var(--intel-text-primary)]"
                            : "border-transparent text-[var(--intel-text-muted)] hover:text-[var(--intel-text-secondary)]"
                        }`}
                >
                    Region Compare{" "}
                    {analyticsCount > 0 && (
                        <span className="ml-1 text-[11px] text-[var(--intel-text-muted)]">({analyticsCount})</span>
                    )}
                </button>
                {selectedRegionName && (
                    <span className="ml-auto text-[11px] text-[var(--intel-text-muted)] px-3 whitespace-nowrap shrink-0">
                        {selectedRegionName}
                    </span>
                )}
            </div>

            {/* Content */}
            <div className="p-4">
                {/* Signal charts */}
                {activeSignal && (
                    <div className="animate-fade-in">
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-[11px] text-[var(--intel-text-muted)]">{activeSignal.description}</span>
                            <span className="text-[11px] text-[var(--intel-text-muted)]">Weekly · 16 weeks</span>
                        </div>
                        <LineChart
                            data={signalCharts[activeSignal.id] ?? []}
                            height={170}
                            strokeColor={activeSignal.strokeColor}
                            fillColor={activeSignal.fillColor}
                        />
                    </div>
                )}

                {tab === "causal" && (
                    <div className="animate-fade-in py-2">
                        <CausalChain />
                    </div>
                )}

                {tab === "compare" && (
                    <div className="animate-fade-in">
                        {comparisonData.length > 0 ? (
                            <ComparisonChart data={comparisonData} />
                        ) : (
                            <div className="flex items-center justify-center h-[170px] text-[13px] text-[var(--intel-text-muted)]">
                                No comparison data available
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

function ComparisonChart({ data }: { data: RegionComparison[] }) {
    const maxScore = Math.max(...data.map((d) => d.riskScore), 1);
    const barW = 56;
    const gap = 10;
    const chartH = 150;
    const totalW = data.length * (barW + gap) - gap;
    const maxBarH = 110;

    return (
        <div className="overflow-x-auto">
            <svg
                viewBox={`0 0 ${Math.max(totalW + 20, 400)} ${chartH + 36}`}
                className="w-full"
                style={{ minWidth: Math.min(totalW + 20, 300) }}
            >
                {data.map((item, idx) => {
                    const barH = Math.max(4, (item.riskScore / maxScore) * maxBarH);
                    const x = idx * (barW + gap);
                    const y = chartH - barH;
                    const color = COMPARISON_COLORS[item.riskLevel] ?? "#C8D8C2";
                    const pct = Math.round(item.riskScore * 100);
                    // Truncate region name to fit under bar
                    const shortName = item.region.length > 12 ? item.region.slice(0, 11) + "…" : item.region;

                    return (
                        <g key={`${item.region}-${idx}`}>
                            <text x={x + barW / 2} y={y - 5} textAnchor="middle" fontSize={10} fill="#374151" fontWeight={600}>
                                {pct}
                            </text>
                            <rect x={x} y={y} width={barW} height={barH} rx={4} fill={color} />
                            <text x={x + barW / 2} y={chartH + 16} textAnchor="middle" fontSize={9} fill="#6B7280">
                                {shortName.split(" ")[0]}
                            </text>
                            <text x={x + barW / 2} y={chartH + 27} textAnchor="middle" fontSize={8} fill="#9CA3AF">
                                {shortName.split(" ").slice(1).join(" ")}
                            </text>
                        </g>
                    );
                })}
                <line x1={0} y1={chartH} x2={totalW} y2={chartH} stroke="#E5E7EB" strokeWidth={1} />
            </svg>
        </div>
    );
}
