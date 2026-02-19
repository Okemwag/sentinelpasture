import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import RiskMap from "@/components/dashboard/risk-map";

const regionalData = [
  {
    region: "Northern Region",
    population: "2.4M",
    stabilityIndex: 72,
    trend: "Stable",
    primaryDriver: "Economic stress",
    confidence: "High",
  },
  {
    region: "Central Region",
    population: "3.8M",
    stabilityIndex: 81,
    trend: "Stable",
    primaryDriver: "Mobility disruption",
    confidence: "Medium",
  },
  {
    region: "Southern Region",
    population: "1.9M",
    stabilityIndex: 67,
    trend: "Moderating",
    primaryDriver: "Incident contagion",
    confidence: "High",
  },
  {
    region: "Eastern District",
    population: "1.2M",
    stabilityIndex: 69,
    trend: "Elevated",
    primaryDriver: "Climate anomaly",
    confidence: "Medium",
  },
];

export default function RegionalRiskPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="Regional Risk Assessment" />

      <Panel title="Regional Distribution">
        <RiskMap />
      </Panel>

      <Panel
        title="Regional Summary"
        metadata={{
          modelVersion: "v2.4.1",
          lastUpdated: "2026-02-19 08:00 UTC",
          confidence: "High",
          showExplainLink: true,
        }}
      >
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#E5E7EB]">
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Region
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Population
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Stability Index
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Trend
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Primary Driver
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Confidence
                </th>
              </tr>
            </thead>
            <tbody>
              {regionalData.map((region, index) => (
                <tr
                  key={index}
                  className="border-b border-[#E5E7EB] last:border-0 hover:bg-[#F9FAFB] transition-colors duration-150"
                >
                  <td className="py-3 px-4 text-[15px] text-[#111111] font-medium">
                    {region.region}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#6B7280]">
                    {region.population}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#111111]">
                    {region.stabilityIndex}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#111111]">
                    {region.trend}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#6B7280]">
                    {region.primaryDriver}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#111111]">
                    {region.confidence}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
