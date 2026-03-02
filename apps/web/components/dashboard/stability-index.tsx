interface StabilityIndexProps {
  value: number;
  trend: "up" | "down" | "stable";
  confidence: "High" | "Medium" | "Low";
  change: string;
}

export default function StabilityIndex({
  value,
  trend,
  confidence,
  change,
}: StabilityIndexProps) {
  const getTrendSymbol = () => {
    if (trend === "up") return "↑";
    if (trend === "down") return "↓";
    return "→";
  };

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
      <div>
        <div className="text-[13px] text-[#6B7280] mb-2">Stability Index</div>
        <div className="text-[28px] sm:text-[32px] font-semibold text-[#111111]">{value}</div>
        <div className="text-[12px] text-[#6B7280] mt-1">Scale: 0–100</div>
      </div>
      
      <div>
        <div className="text-[13px] text-[#6B7280] mb-2">Trend</div>
        <div className="text-[28px] sm:text-[32px] font-normal text-[#111111]">
          {getTrendSymbol()}
        </div>
        <div className="text-[12px] text-[#6B7280] mt-1">Direction</div>
      </div>
      
      <div>
        <div className="text-[13px] text-[#6B7280] mb-2">Confidence</div>
        <div className="text-[16px] sm:text-[18px] font-medium text-[#111111] mt-2 sm:mt-3">
          {confidence}
        </div>
      </div>
      
      <div>
        <div className="text-[13px] text-[#6B7280] mb-2">Change</div>
        <div className="text-[16px] sm:text-[18px] font-medium text-[#111111] mt-2 sm:mt-3">
          {change}
        </div>
        <div className="text-[12px] text-[#6B7280] mt-1">vs previous period</div>
      </div>
    </div>
  );
}
