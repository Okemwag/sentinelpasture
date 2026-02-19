import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";

interface Intervention {
  category: string;
  expectedImpact: string;
  timeToEffect: string;
  costBand: string;
  confidence: string;
}

const interventions: Intervention[] = [
  {
    category: "Economic stabilization program",
    expectedImpact: "High",
    timeToEffect: "Medium",
    costBand: "$$$$",
    confidence: "High",
  },
  {
    category: "Infrastructure resilience enhancement",
    expectedImpact: "Moderate",
    timeToEffect: "Long",
    costBand: "$$$$$",
    confidence: "Medium",
  },
  {
    category: "Community engagement initiative",
    expectedImpact: "Moderate",
    timeToEffect: "Short",
    costBand: "$$",
    confidence: "High",
  },
  {
    category: "Education system support",
    expectedImpact: "Low",
    timeToEffect: "Long",
    costBand: "$$$",
    confidence: "Medium",
  },
];

export default function InterventionsPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="Intervention Analysis" />

      <div className="bg-[#F9FAFB] border border-[#E5E7EB] rounded-[8px] p-4">
        <p className="text-[13px] text-[#6B7280] leading-relaxed">
          Analysis region: Southern Region. Recommendations based on current risk
          profile and historical intervention effectiveness. Final decision
          authority remains with designated governance bodies.
        </p>
      </div>

      <Panel
        title="Ranked Intervention Options"
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
                  Intervention Category
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Expected Impact
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Time to Effect
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Cost Band
                </th>
                <th className="text-left py-3 px-4 text-[13px] font-medium text-[#6B7280]">
                  Confidence
                </th>
              </tr>
            </thead>
            <tbody>
              {interventions.map((intervention, index) => (
                <tr
                  key={index}
                  className="border-b border-[#E5E7EB] last:border-0 hover:bg-[#F9FAFB] transition-colors duration-150"
                >
                  <td className="py-3 px-4 text-[15px] text-[#111111]">
                    {intervention.category}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#111111]">
                    {intervention.expectedImpact}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#6B7280]">
                    {intervention.timeToEffect}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#6B7280]">
                    {intervention.costBand}
                  </td>
                  <td className="py-3 px-4 text-[15px] text-[#111111]">
                    {intervention.confidence}
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
