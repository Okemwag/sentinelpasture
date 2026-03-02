import PageHeader from "@/components/dashboard/page-header";
import DateRangeSelector from "@/components/dashboard/date-range-selector";
import Panel from "@/components/dashboard/panel";
import StabilityIndex from "@/components/dashboard/stability-index";
import RiskMap from "@/components/dashboard/risk-map";
import PressureZonesTable from "@/components/dashboard/pressure-zones-table";
import { Icon } from "@iconify/react";

export default function DashboardOverview() {
  return (
    <div className="space-y-6">
      <PageHeader title="National Stability Overview">
        <DateRangeSelector />
      </PageHeader>

      {/* Quick Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[13px] text-[#6B7280]">Active Alerts</span>
            <Icon icon="mdi:alert-circle-outline" className="w-5 h-5 text-[#6B7280]" />
          </div>
          <div className="text-[24px] font-semibold text-[#111111]">3</div>
          <div className="text-[12px] text-[#6B7280] mt-1">Requires attention</div>
        </div>

        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[13px] text-[#6B7280]">Regions Monitored</span>
            <Icon icon="mdi:map-marker-multiple-outline" className="w-5 h-5 text-[#6B7280]" />
          </div>
          <div className="text-[24px] font-semibold text-[#111111]">47</div>
          <div className="text-[12px] text-[#6B7280] mt-1">Counties tracked</div>
        </div>

        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[13px] text-[#6B7280]">Data Sources</span>
            <Icon icon="mdi:database-outline" className="w-5 h-5 text-[#6B7280]" />
          </div>
          <div className="text-[24px] font-semibold text-[#111111]">24</div>
          <div className="text-[12px] text-[#6B7280] mt-1">Active feeds</div>
        </div>

        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[13px] text-[#6B7280]">Last Update</span>
            <Icon icon="mdi:clock-outline" className="w-5 h-5 text-[#6B7280]" />
          </div>
          <div className="text-[24px] font-semibold text-[#111111]">2m</div>
          <div className="text-[12px] text-[#6B7280] mt-1">ago</div>
        </div>
      </div>

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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Panel title="Regional Risk Distribution">
          <RiskMap />
        </Panel>

        <Panel title="Risk Factors Summary">
          <div className="space-y-4">
            <div className="flex items-start gap-3">
              <Icon icon="mdi:trending-up" className="w-5 h-5 text-[#6B7280] mt-0.5 flex-shrink-0" />
              <div>
                <div className="text-[15px] font-medium text-[#111111] mb-1">
                  Economic Indicators
                </div>
                <div className="text-[13px] text-[#6B7280] leading-relaxed">
                  Inflation at 6.8%, unemployment stable at 5.2%. Youth unemployment remains elevated at 12.4%.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Icon icon="mdi:weather-partly-cloudy" className="w-5 h-5 text-[#6B7280] mt-0.5 flex-shrink-0" />
              <div>
                <div className="text-[15px] font-medium text-[#111111] mb-1">
                  Climate Patterns
                </div>
                <div className="text-[13px] text-[#6B7280] leading-relaxed">
                  Rainfall 15% below seasonal average. Drought conditions in 8 counties.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Icon icon="mdi:account-group-outline" className="w-5 h-5 text-[#6B7280] mt-0.5 flex-shrink-0" />
              <div>
                <div className="text-[15px] font-medium text-[#111111] mb-1">
                  Social Cohesion
                </div>
                <div className="text-[13px] text-[#6B7280] leading-relaxed">
                  Community trust index at 68/100. Slight decline in civic participation noted.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3">
              <Icon icon="mdi:hospital-building" className="w-5 h-5 text-[#6B7280] mt-0.5 flex-shrink-0" />
              <div>
                <div className="text-[15px] font-medium text-[#111111] mb-1">
                  Public Health
                </div>
                <div className="text-[13px] text-[#6B7280] leading-relaxed">
                  Healthcare capacity at 78%. No major disease outbreaks reported.
                </div>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <Panel title="Top Emerging Pressure Zones">
        <PressureZonesTable />
      </Panel>
    </div>
  );
}
