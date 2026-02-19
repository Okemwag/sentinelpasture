import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import { Icon } from "@iconify/react";

const alerts = [
  {
    id: 1,
    severity: "elevated",
    title: "Economic stress indicators rising in Turkana County",
    description: "Unemployment rate increased by 2.3% over the past 30 days. Food prices up 8.4%.",
    timestamp: "2026-02-19 06:30 UTC",
    region: "Turkana County",
    status: "active",
  },
  {
    id: 2,
    severity: "moderate",
    title: "Drought conditions persisting in Northern regions",
    description: "Rainfall 22% below seasonal average. Water scarcity affecting 3 counties.",
    timestamp: "2026-02-18 14:15 UTC",
    region: "Northern Region",
    status: "active",
  },
  {
    id: 3,
    severity: "elevated",
    title: "Youth unemployment spike in Nairobi County",
    description: "Youth unemployment reached 14.2%, up from 12.1% last quarter.",
    timestamp: "2026-02-17 09:45 UTC",
    region: "Nairobi County",
    status: "monitoring",
  },
  {
    id: 4,
    severity: "moderate",
    title: "Healthcare capacity strain in Mombasa",
    description: "Hospital bed occupancy at 89%. Staffing levels below recommended threshold.",
    timestamp: "2026-02-16 11:20 UTC",
    region: "Mombasa County",
    status: "resolved",
  },
];

export default function AlertsPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="National Alerts & Notifications" />

      {/* Alert Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[13px] text-[#6B7280]">Active Alerts</span>
            <Icon icon="mdi:alert-circle-outline" className="w-5 h-5 text-[#6B7280]" />
          </div>
          <div className="text-[24px] font-semibold text-[#111111]">2</div>
          <div className="text-[12px] text-[#6B7280] mt-1">Requires attention</div>
        </div>

        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[13px] text-[#6B7280]">Monitoring</span>
            <Icon icon="mdi:eye-outline" className="w-5 h-5 text-[#6B7280]" />
          </div>
          <div className="text-[24px] font-semibold text-[#111111]">1</div>
          <div className="text-[12px] text-[#6B7280] mt-1">Under observation</div>
        </div>

        <div className="bg-white rounded-[8px] border border-[#E5E7EB] p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[13px] text-[#6B7280]">Resolved (24h)</span>
            <Icon icon="mdi:check-circle-outline" className="w-5 h-5 text-[#6B7280]" />
          </div>
          <div className="text-[24px] font-semibold text-[#111111]">1</div>
          <div className="text-[12px] text-[#6B7280] mt-1">Last day</div>
        </div>
      </div>

      <Panel title="Recent Alerts">
        <div className="space-y-4">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className="border border-[#E5E7EB] rounded-[8px] p-4 hover:bg-[#F9FAFB] transition-colors duration-150"
            >
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 mt-1">
                  {alert.severity === "elevated" && (
                    <Icon icon="mdi:alert-circle-outline" className="w-6 h-6 text-[#6B7280]" />
                  )}
                  {alert.severity === "moderate" && (
                    <Icon icon="mdi:information-outline" className="w-6 h-6 text-[#6B7280]" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <h3 className="text-[15px] font-medium text-[#111111]">
                      {alert.title}
                    </h3>
                    <span
                      className={`
                        flex-shrink-0 px-2 py-1 text-[12px] rounded-[4px]
                        ${
                          alert.status === "active"
                            ? "bg-[#F3F4F6] text-[#111111]"
                            : alert.status === "monitoring"
                            ? "bg-[#F9FAFB] text-[#6B7280]"
                            : "bg-white border border-[#E5E7EB] text-[#6B7280]"
                        }
                      `}
                    >
                      {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                    </span>
                  </div>

                  <p className="text-[13px] text-[#6B7280] leading-relaxed mb-3">
                    {alert.description}
                  </p>

                  <div className="flex flex-wrap items-center gap-4 text-[12px] text-[#6B7280]">
                    <div className="flex items-center gap-1">
                      <Icon icon="mdi:map-marker-outline" className="w-4 h-4" />
                      <span>{alert.region}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Icon icon="mdi:clock-outline" className="w-4 h-4" />
                      <span>{alert.timestamp}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Alert Configuration">
        <div className="space-y-4">
          <div>
            <label className="block text-[13px] text-[#6B7280] mb-2">
              Alert Threshold
            </label>
            <select
              className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111]"
              defaultValue="moderate"
            >
              <option value="all">All alerts</option>
              <option value="moderate">Moderate and above</option>
              <option value="elevated">Elevated only</option>
            </select>
          </div>

          <div>
            <label className="block text-[13px] text-[#6B7280] mb-2">
              Notification Method
            </label>
            <div className="space-y-2">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  defaultChecked
                  className="w-4 h-4 rounded border-[#E5E7EB]"
                />
                <span className="text-[15px] text-[#111111]">Dashboard notifications</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  defaultChecked
                  className="w-4 h-4 rounded border-[#E5E7EB]"
                />
                <span className="text-[15px] text-[#111111]">Email alerts</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="w-4 h-4 rounded border-[#E5E7EB]"
                />
                <span className="text-[15px] text-[#111111]">SMS notifications</span>
              </label>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
}
