"use client";

import { useState } from "react";
import LineChart from "@/components/dashboard/line-chart";
import CausalChain from "@/components/dashboard/causal-chain";

interface ChartPoint { date: string; value: number; }
interface RegionComparson { region: string; riskScore: number; riskLevel: string; }

interface AnalyticsStripProps {
    trendData: ChartPoint[];
    comparisonData: RegionComparson[];
    selectedRegionName?: string;
}

type Tab = "trend" | "causal" | "compare";

const COMPARISON_COLORS: Record<string, string> = {
    low: "#C8D8C2",
    watch: "#C7A56B",
    elevated: "#8C6A3D",
    critical: "#4A3A26",
};

export default function AnalyticsStrip({ trendData, comparisonData, selectedRegionName }: AnalyticsStripProps) {
    const [tab, setTab] = useState<Tab>("trend");

    const tabs: { id: Tab; label: string }[] = [
        { id: "trend", label: "Risk Trend" },
        { id: "causal", label: "Causal Chain" },
        { id: "compare", label: "Regional Comparison" },
    ];

    return (
        <div className="bg-white border border-[#E5E7EB] rounded-[8px]">
            {/* Tab header */}
            <div className="flex items-center border-b border-[#E5E7EB] px-4">
                {tabs.map((t) => (
                    <button
                        key={t.id}
                        type="button"
                        onClick={() => setTab(t.id)}
                        className={`px-4 py-3 text-[13px] font-medium border-b-2 transition-colors mr-1 ${tab === t.id
                                ? "border-[#111111] text-[#111111]"
                                : "border-transparent text-[#6B7280] hover:text-[#374151]"
                            }`}
                    >
                        {t.label}
                    </button>
                ))}
                {selectedRegionName && (
                    <span className="ml-auto text-[12px] text-[#9CA3AF] pr-2">
                        {selectedRegionName}
                    </span>
                )}
            </div>

            {/* Tab content */}
            <div className="p-4">
                {tab === "trend" && (
                    <div className="animate-fade-in">
                        {trendData.length > 0 ? (
                            <LineChart data={trendData} height={180} />
                        ) : (
                            <div className="flex items-center justify-center h-[180px] text-[13px] text-[#9CA3AF]">
                                Select a region to view risk trend
                            </div>
                        )}
                    </div>
                )}

                {tab === "causal" && (
                    <div className="animate-fade-in py-4">
                        <CausalChain />
                    </div>
                )}

                {tab === "compare" && (
                    <div className="animate-fade-in">
                        {comparisonData.length > 0 ? (
                            <ComparisonChart data={comparisonData} />
                        ) : (
                            <div className="flex items-center justify-center h-[180px] text-[13px] text-[#9CA3AF]">
                                No comparison data available
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

function ComparisonChart({ data }: { data: RegionComparson[] }) {
    const maxScore = Math.max(...data.map((d) => d.riskScore), 1);
    const barW = Math.floor(Math.min(80, 600 / data.length));
    const gap = 8;
    const chartH = 160;
    const totalW = data.length * (barW + gap) - gap;
    const maxBarH = 120;

    return (
        <div className="overflow-x-auto">
            <svg
                viewBox={`0 0 ${Math.max(totalW, 400)} ${chartH + 32}`}
                className="w-full"
                style={{ minWidth: Math.min(totalW, 300) }}
            >
                {data.map((item, idx) => {
                    const barH = (item.riskScore / maxScore) * maxBarH;
                    const x = idx * (barW + gap);
                    const y = chartH - barH;
                    const color = COMPARISON_COLORS[item.riskLevel] ?? "#C8D8C2";
                    const pct = Math.round(item.riskScore * 100);

                    return (
                        <g key={item.region}>
                            {/* Risk score label on top */}
                            <text
                                x={x + barW / 2}
                                y={y - 4}
                                textAnchor="middle"
                                fontSize={10}
                                fill="#374151"
                                fontWeight={600}
                            >
                                {pct}
                            </text>
                            {/* Bar */}
                            <rect x={x} y={y} width={barW} height={barH} rx={4} fill={color} />
                            {/* Region label */}
                            <text
                                x={x + barW / 2}
                                y={chartH + 16}
                                textAnchor="middle"
                                fontSize={9}
                                fill="#6B7280"
                            >
                                {item.region.length > 10 ? item.region.slice(0, 9) + "…" : item.region}
                            </text>
                        </g>
                    );
                })}
                {/* Baseline */}
                <line
                    x1={0}
                    y1={chartH}
                    x2={totalW}
                    y2={chartH}
                    stroke="#E5E7EB"
                    strokeWidth={1}
                />
            </svg>
        </div>
    );
}
