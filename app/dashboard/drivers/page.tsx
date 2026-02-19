import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import DriverBar from "@/components/dashboard/driver-bar";

const drivers = [
  {
    label: "Economic stress",
    percentage: 34,
    trend: "up" as const,
    confidence: "High",
  },
  {
    label: "Climate anomaly",
    percentage: 28,
    trend: "up" as const,
    confidence: "Medium",
  },
  {
    label: "Mobility disruption",
    percentage: 18,
    trend: "stable" as const,
    confidence: "High",
  },
  {
    label: "Education attendance decline",
    percentage: 12,
    trend: "down" as const,
    confidence: "Medium",
  },
  {
    label: "Incident contagion",
    percentage: 8,
    trend: "up" as const,
    confidence: "High",
  },
];

export default function DriversPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="Risk Drivers" />

      <Panel
        title="Driver Breakdown"
        metadata={{
          modelVersion: "v2.4.1",
          lastUpdated: "2026-02-19 08:00 UTC",
          confidence: "High",
          showExplainLink: true,
        }}
      >
        <div className="space-y-4">
          {drivers.map((driver) => (
            <DriverBar key={driver.label} {...driver} />
          ))}
        </div>
      </Panel>

      <Panel title="Driver Definitions">
        <div className="space-y-4 text-[15px] leading-relaxed">
          <div>
            <div className="font-medium text-[#111111] mb-1">Economic stress</div>
            <div className="text-[#6B7280]">
              Composite measure of unemployment rates, inflation indices, and
              household income volatility across monitored regions.
            </div>
          </div>
          <div>
            <div className="font-medium text-[#111111] mb-1">Climate anomaly</div>
            <div className="text-[#6B7280]">
              Deviation from historical climate patterns including temperature,
              precipitation, and extreme weather event frequency.
            </div>
          </div>
          <div>
            <div className="font-medium text-[#111111] mb-1">
              Mobility disruption
            </div>
            <div className="text-[#6B7280]">
              Changes in population movement patterns, transportation
              accessibility, and cross-regional connectivity.
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
}
