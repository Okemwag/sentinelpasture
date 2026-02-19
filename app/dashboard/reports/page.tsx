import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";

const reports = [
  {
    title: "National Stability Assessment",
    type: "PDF Report",
    date: "2026-02-15",
    size: "2.4 MB",
  },
  {
    title: "Regional Risk Data Export",
    type: "CSV Export",
    date: "2026-02-15",
    size: "156 KB",
  },
  {
    title: "Intervention Effectiveness Analysis",
    type: "PDF Report",
    date: "2026-02-01",
    size: "1.8 MB",
  },
  {
    title: "System Audit Log",
    type: "CSV Export",
    date: "2026-02-01",
    size: "89 KB",
  },
];

export default function ReportsPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="Reports & Exports" />

      <Panel title="Available Reports">
        <div className="space-y-3">
          {reports.map((report, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-4 border border-[#E5E7EB] rounded-[8px] hover:bg-[#F9FAFB] transition-colors duration-150"
            >
              <div>
                <div className="text-[15px] text-[#111111] font-medium mb-1">
                  {report.title}
                </div>
                <div className="text-[13px] text-[#6B7280]">
                  {report.type} • {report.date} • {report.size}
                </div>
              </div>
              <button className="px-4 py-2 text-[13px] text-[#374151] border border-[#E5E7EB] rounded-[8px] hover:bg-[#F3F4F6] transition-colors duration-150">
                Download
              </button>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Generate New Report">
        <div className="space-y-4">
          <div>
            <label className="block text-[13px] text-[#6B7280] mb-2">
              Report Type
            </label>
            <select className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111]">
              <option>National Stability Assessment</option>
              <option>Regional Risk Analysis</option>
              <option>Driver Breakdown</option>
              <option>Intervention Summary</option>
            </select>
          </div>

          <div>
            <label className="block text-[13px] text-[#6B7280] mb-2">
              Date Range
            </label>
            <select className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111]">
              <option>Last 7 days</option>
              <option>Last 30 days</option>
              <option>Last 90 days</option>
              <option>Custom range</option>
            </select>
          </div>

          <div>
            <label className="block text-[13px] text-[#6B7280] mb-2">
              Format
            </label>
            <select className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111]">
              <option>PDF</option>
              <option>CSV</option>
            </select>
          </div>

          <button className="px-4 py-2 text-[13px] bg-[#374151] text-white rounded-[8px] hover:bg-[#1F2937] transition-colors duration-150">
            Generate Report
          </button>
        </div>
      </Panel>
    </div>
  );
}
