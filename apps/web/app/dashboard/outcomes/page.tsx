import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import LineChart from "@/components/dashboard/line-chart";

const outcomeData = [
  {
    intervention: "Economic stabilization program",
    deployed: "2025-11-15",
    riskBefore: 78,
    riskAfter: 67,
    trend: "Moderating",
    commentary:
      "Initial indicators suggest reduced economic stress in target region. Continued monitoring required.",
  },
  {
    intervention: "Community engagement initiative",
    deployed: "2025-12-01",
    riskBefore: 72,
    riskAfter: 69,
    trend: "Stable",
    commentary:
      "Modest improvement observed. Long-term effectiveness assessment pending.",
  },
];

const chartData = [
  { date: "Nov 1", value: 78 },
  { date: "Nov 8", value: 76 },
  { date: "Nov 15", value: 74 },
  { date: "Nov 22", value: 71 },
  { date: "Nov 29", value: 69 },
  { date: "Dec 6", value: 68 },
  { date: "Dec 13", value: 67 },
];

export default function OutcomesPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="Intervention Outcomes" />

      <Panel title="Risk Trend Following Intervention">
        <LineChart data={chartData} height={240} />
        <div className="mt-4 text-[13px] text-[#6B7280]">
          Southern Region stability index over time. Intervention deployed November 15.
        </div>
      </Panel>

      <Panel
        title="Deployed Interventions"
        metadata={{
          modelVersion: "v2.4.1",
          lastUpdated: "2026-02-19 08:00 UTC",
        }}
      >
        <div className="space-y-6">
          {outcomeData.map((outcome, index) => (
            <div
              key={index}
              className="pb-6 last:pb-0 border-b border-[#E5E7EB] last:border-0"
            >
              <div className="grid grid-cols-2 gap-6 mb-4">
                <div>
                  <div className="text-[13px] text-[#6B7280] mb-1">
                    Intervention
                  </div>
                  <div className="text-[15px] text-[#111111] font-medium">
                    {outcome.intervention}
                  </div>
                </div>
                <div>
                  <div className="text-[13px] text-[#6B7280] mb-1">
                    Date Deployed
                  </div>
                  <div className="text-[15px] text-[#111111]">
                    {outcome.deployed}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-6 mb-4">
                <div>
                  <div className="text-[13px] text-[#6B7280] mb-1">
                    Risk Before
                  </div>
                  <div className="text-[18px] text-[#111111] font-medium">
                    {outcome.riskBefore}
                  </div>
                </div>
                <div>
                  <div className="text-[13px] text-[#6B7280] mb-1">
                    Risk After
                  </div>
                  <div className="text-[18px] text-[#111111] font-medium">
                    {outcome.riskAfter}
                  </div>
                </div>
                <div>
                  <div className="text-[13px] text-[#6B7280] mb-1">
                    Observed Trend
                  </div>
                  <div className="text-[15px] text-[#111111]">
                    {outcome.trend}
                  </div>
                </div>
                <div>
                  <div className="text-[13px] text-[#6B7280] mb-1">Change</div>
                  <div className="text-[18px] text-[#111111] font-medium">
                    {outcome.riskAfter - outcome.riskBefore}
                  </div>
                </div>
              </div>

              <div>
                <div className="text-[13px] text-[#6B7280] mb-1">Commentary</div>
                <div className="text-[15px] text-[#111111] leading-relaxed">
                  {outcome.commentary}
                </div>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
