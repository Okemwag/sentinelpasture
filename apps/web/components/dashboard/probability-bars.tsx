"use client";

interface ProbabilityItem {
    label: string;
    probability: number; // 0–100
}

interface ProbabilityBarsProps {
    items: ProbabilityItem[];
}

export default function ProbabilityBars({ items }: ProbabilityBarsProps) {
    return (
        <div className="space-y-3">
            {items.map((item) => {
                const pct = Math.min(100, Math.max(0, item.probability));
                const color =
                    pct >= 70 ? "#4A3A26" : pct >= 50 ? "#8C6A3D" : pct >= 30 ? "#C7A56B" : "#7A9B70";
                const bgColor =
                    pct >= 70 ? "#EDE8E0" : pct >= 50 ? "#F5EFE6" : pct >= 30 ? "#FBF5EB" : "#EDF4EB";
                return (
                    <div key={item.label}>
                        <div className="mb-1.5 flex items-center justify-between">
                            <span className="text-[13px] text-[#374151]">{item.label}</span>
                            <span className="text-[13px] font-semibold" style={{ color }}>
                                {pct}%
                            </span>
                        </div>
                        <div
                            className="h-[6px] w-full overflow-hidden rounded-full"
                            style={{ backgroundColor: bgColor }}
                        >
                            <div
                                className="h-full rounded-full animate-grow-bar"
                                style={{ width: `${pct}%`, backgroundColor: color }}
                            />
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
