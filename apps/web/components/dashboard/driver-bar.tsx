interface DriverBarProps {
  label: string;
  percentage: number;
  trend: "up" | "down" | "stable";
  confidence: string;
}

export default function DriverBar({
  label,
  percentage,
  trend,
  confidence,
}: DriverBarProps) {
  const getTrendSymbol = () => {
    if (trend === "up") return "↑";
    if (trend === "down") return "↓";
    return "→";
  };

  return (
    <div className="mb-6 last:mb-0">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[15px] text-[#111111]">{label}</span>
        <div className="flex items-center gap-4 text-[13px]">
          <span className="text-[#6B7280]">{getTrendSymbol()}</span>
          <span className="text-[#111111] font-medium">{percentage}%</span>
          <span className="text-[#6B7280]">{confidence}</span>
        </div>
      </div>
      <div className="h-2 bg-[#F3F4F6] rounded-[2px] overflow-hidden">
        <div
          className="h-full bg-[#6B7280] transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
