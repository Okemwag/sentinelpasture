interface PressureZone {
  region: string;
  riskTrend: string;
  primaryDriver: string;
  secondaryDriver: string;
  confidence: string;
}

const mockData: PressureZone[] = [
  {
    region: "Southern Region",
    riskTrend: "Increasing",
    primaryDriver: "Incident contagion",
    secondaryDriver: "Economic stress",
    confidence: "High",
  },
  {
    region: "Eastern District",
    riskTrend: "Elevated",
    primaryDriver: "Climate anomaly",
    secondaryDriver: "Mobility disruption",
    confidence: "Medium",
  },
  {
    region: "Western Province",
    riskTrend: "Moderating",
    primaryDriver: "Education decline",
    secondaryDriver: "Economic stress",
    confidence: "High",
  },
];

export default function PressureZonesTable() {
  return (
    <div className="overflow-x-auto -mx-6 px-6">
      <table className="w-full min-w-[640px]">
        <thead>
          <tr className="border-b border-[#E5E7EB]">
            <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
              Region
            </th>
            <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
              Risk Trend
            </th>
            <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
              Primary Driver
            </th>
            <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
              Secondary Driver
            </th>
            <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
              Confidence
            </th>
          </tr>
        </thead>
        <tbody>
          {mockData.map((zone, index) => (
            <tr
              key={index}
              className="border-b border-[#E5E7EB] last:border-0 hover:bg-[#F9FAFB] transition-colors duration-150"
            >
              <td className="py-3 px-4 text-[15px] text-[#111111]">
                {zone.region}
              </td>
              <td className="py-3 px-4 text-[15px] text-[#111111]">
                {zone.riskTrend}
              </td>
              <td className="py-3 px-4 text-[15px] text-[#6B7280]">
                {zone.primaryDriver}
              </td>
              <td className="py-3 px-4 text-[15px] text-[#6B7280]">
                {zone.secondaryDriver}
              </td>
              <td className="py-3 px-4 text-[15px] text-[#111111]">
                {zone.confidence}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
