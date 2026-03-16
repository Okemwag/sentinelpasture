"use client";

import { useState } from "react";

interface Intervention {
    category: string;
    expectedImpact: string;
    timeToEffect: string;
    costBand: string;
    confidence: string;
}

interface InterventionCardsProps {
    interventions: Intervention[];
    context?: {
        regionName?: string;
        thresholdStatus?: string;
        primaryDriver?: string;
        thresholdReason?: string;
    };
}

const RANK_LABELS = ["1st Priority", "2nd Priority", "3rd Priority", "4th Priority"];

function costColor(band: string) {
    const b = band.toLowerCase();
    if (b.includes("low")) return { bg: "#EDF4EB", text: "#3A6B33", border: "#A8C9A2" };
    if (b.includes("medium") || b.includes("moderate")) return { bg: "#FBF5EB", text: "#7A4F1E", border: "#E2C99A" };
    return { bg: "#EDE8E0", text: "#2A1E12", border: "#BFB099" };
}

function confColor(conf: string) {
    const c = conf.toLowerCase();
    if (c === "high") return "#3A6B33";
    if (c === "medium") return "#8C6A3D";
    return "#6B7280";
}

export default function InterventionCards({ interventions, context }: InterventionCardsProps) {
    const [expanded, setExpanded] = useState<string | null>(interventions[0]?.category ?? null);

    if (!interventions.length) {
        return <div className="text-[13px] text-[#6B7280] py-4">No interventions available.</div>;
    }

    return (
        <div className="space-y-2">
            {interventions.slice(0, 5).map((item, idx) => {
                const isOpen = expanded === item.category;
                const cost = costColor(item.costBand);
                return (
                    <div
                        key={`${item.category}-${idx}`}
                        className="rounded-[8px] border border-[#E5E7EB] overflow-hidden"
                    >
                        <button
                            type="button"
                            onClick={() => setExpanded(isOpen ? null : item.category)}
                            className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left hover:bg-[#FAFAFA] transition-colors"
                        >
                            <div className="flex items-center gap-3 min-w-0">
                                <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#6B7280] w-20 shrink-0 text-right">
                                    {RANK_LABELS[idx] ?? `#${idx + 1}`}
                                </span>
                                <div className="min-w-0">
                                    <div className="text-[14px] font-medium text-[#111111] truncate">
                                        {item.category}
                                    </div>
                                    <div className="text-[11px] text-[#6B7280] truncate">
                                        {context?.regionName
                                            ? `For ${context.regionName}${context.thresholdStatus ? ` — ${context.thresholdStatus}` : ""}`
                                            : "For the current highest modelled pressure region"}
                                    </div>
                                </div>
                            </div>
                            <div className="flex items-center gap-2 shrink-0">
                                <span
                                    className="text-[11px] px-2 py-0.5 rounded-full border font-medium"
                                    style={{ backgroundColor: cost.bg, color: cost.text, borderColor: cost.border }}
                                >
                                    {item.costBand}
                                </span>
                                <svg
                                    className={`h-4 w-4 text-[#9CA3AF] transition-transform ${isOpen ? "rotate-180" : ""}`}
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                    strokeWidth={2}
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                                </svg>
                            </div>
                        </button>

                        {isOpen && (
                            <div className="px-4 pb-4 pt-2 border-t border-[#F3F4F6] bg-[#FAFAFA] animate-fade-in">
                                <div className="grid grid-cols-2 gap-4 mb-3">
                                    <div>
                                        <div className="text-[11px] uppercase tracking-[0.14em] text-[#9CA3AF] mb-1">Expected Impact</div>
                                        <div className="text-[13px] text-[#111111]">{item.expectedImpact}</div>
                                    </div>
                                    <div>
                                        <div className="text-[11px] uppercase tracking-[0.14em] text-[#9CA3AF] mb-1">Time to Effect</div>
                                        <div className="text-[13px] text-[#111111]">{item.timeToEffect}</div>
                                    </div>
                                </div>
                                {context?.regionName && (
                                    <div className="mt-3 grid grid-cols-1 gap-1.5 text-[12px] text-[#4B5563]">
                                        <div>
                                            <span className="text-[11px] uppercase tracking-[0.14em] text-[#9CA3AF] mr-1">
                                                Region
                                            </span>
                                            <span className="font-medium text-[#111111]">
                                                {context.regionName}
                                            </span>
                                        </div>
                                        {context.thresholdStatus && (
                                            <div>
                                                <span className="text-[11px] uppercase tracking-[0.14em] text-[#9CA3AF] mr-1">
                                                    Why Now
                                                </span>
                                                <span>{context.thresholdStatus}</span>
                                            </div>
                                        )}
                                        {context.primaryDriver && (
                                            <div>
                                                <span className="text-[11px] uppercase tracking-[0.14em] text-[#9CA3AF] mr-1">
                                                    Primary Driver
                                                </span>
                                                <span>{context.primaryDriver}</span>
                                            </div>
                                        )}
                                    </div>
                                )}
                                <div className="flex items-center gap-1 mt-2">
                                    <span className="text-[11px] text-[#9CA3AF] uppercase tracking-[0.14em]">Confidence:</span>
                                    <span
                                        className="text-[12px] font-semibold ml-1"
                                        style={{ color: confColor(item.confidence) }}
                                    >
                                        {item.confidence}
                                    </span>
                                </div>
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
