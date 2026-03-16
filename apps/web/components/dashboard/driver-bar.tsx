interface DriverBarProps {
  label: string;
  percentage: number;
  trend: "up" | "down" | "stable";
  confidence: string;
}

export default function DriverBar({ label, percentage, trend, confidence }: DriverBarProps) {
  const pct = Math.min(100, Math.max(0, percentage));
  // Color by impact tier
  const { barColor, labelColor, bgColor } =
    pct >= 70
      ? { barColor: "#4A3A26", labelColor: "#4A3A26", bgColor: "#EDE8E0" }
      : pct >= 50
        ? { barColor: "#8C6A3D", labelColor: "#8C6A3D", bgColor: "#F5EFE6" }
        : pct >= 30
          ? { barColor: "#C7A56B", labelColor: "#9A6E2D", bgColor: "#FBF5EB" }
          : { barColor: "#7A9B70", labelColor: "#3A6B33", bgColor: "#EDF4EB" };

  const trendArrow = trend === "up" ? "↑" : trend === "down" ? "↓" : "→";
  const trendColor = trend === "up" ? "#8C6A3D" : trend === "down" ? "#3A6B33" : "#9CA3AF";

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[13px] font-medium text-[#111111]">{label}</span>
        <div className="flex items-center gap-3">
          <span className="text-[12px]" style={{ color: trendColor }}>{trendArrow}</span>
          <span
            className="text-[11px] font-semibold px-2 py-0.5 rounded-full"
            style={{ backgroundColor: bgColor, color: labelColor }}
          >
            {pct.toFixed(0)}%
          </span>
          <span className="text-[11px] text-[#9CA3AF] w-12 text-right">{confidence}</span>
        </div>
      </div>
      <div className="h-[6px] rounded-full overflow-hidden" style={{ backgroundColor: "#F3F4F6" }}>
        <div
          className="h-full rounded-full animate-grow-bar"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}
