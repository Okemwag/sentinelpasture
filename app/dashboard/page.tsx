import PageHeader from "@/components/dashboard/page-header";
import DateRangeSelector from "@/components/dashboard/date-range-selector";
import Panel from "@/components/dashboard/panel";
import StabilityIndex from "@/components/dashboard/stability-index";
import RiskMap from "@/components/dashboard/risk-map";
import PressureZonesTable from "@/components/dashboard/pressure-zones-table";

export default function DashboardOverview() {
  return (
    <div className="space-y-6">
      <PageHeader title="National Stability Overview">
        <DateRangeSelector />
      </PageHeader>

      <Panel
        title="National Risk Summary"
        metadata={{
          modelVersion: "v2.4.1",
          lastUpdated: "2026-02-19 08:00 UTC",
          confidence: "High",
          showExplainLink: true,
        }}
      >
        <StabilityIndex
          value={67}
          trend="down"
          confidence="High"
          change="-3.2 points"
        />
      </Panel>

      <Panel title="Regional Risk Distribution">
        <RiskMap />
      </Panel>

      <Panel title="Top Emerging Pressure Zones">
        <PressureZonesTable />
      </Panel>
    </div>
  );
}
