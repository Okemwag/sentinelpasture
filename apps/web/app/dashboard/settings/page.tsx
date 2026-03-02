import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <PageHeader title="Settings" />

      <Panel title="System Information">
        <div className="space-y-4 text-[15px]">
          <div className="flex justify-between py-2 border-b border-[#E5E7EB]">
            <span className="text-[#6B7280]">Platform Version</span>
            <span className="text-[#111111]">2.4.1</span>
          </div>
          <div className="flex justify-between py-2 border-b border-[#E5E7EB]">
            <span className="text-[#6B7280]">Model Version</span>
            <span className="text-[#111111]">v2.4.1</span>
          </div>
          <div className="flex justify-between py-2 border-b border-[#E5E7EB]">
            <span className="text-[#6B7280]">Last System Update</span>
            <span className="text-[#111111]">2026-02-15 14:30 UTC</span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-[#6B7280]">Data Sources</span>
            <span className="text-[#111111]">12 active</span>
          </div>
        </div>
      </Panel>

      <Panel title="Display Preferences">
        <div className="space-y-4">
          <div>
            <label className="block text-[13px] text-[#6B7280] mb-2">
              Default Time Range
            </label>
            <select 
              className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111]"
              defaultValue="30d"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
            </select>
          </div>

          <div>
            <label className="block text-[13px] text-[#6B7280] mb-2">
              Confidence Threshold
            </label>
            <select 
              className="w-full px-3 py-2 text-[15px] border border-[#E5E7EB] rounded-[8px] bg-white text-[#111111]"
              defaultValue="medium"
            >
              <option value="all">Show all</option>
              <option value="medium">Medium and above</option>
              <option value="high">High only</option>
            </select>
          </div>
        </div>
      </Panel>

      <Panel title="Data Management">
        <div className="space-y-3">
          <button className="w-full px-4 py-3 text-[15px] text-left border border-[#E5E7EB] rounded-[8px] hover:bg-[#F9FAFB] transition-colors duration-150">
            Export all data
          </button>
          <button className="w-full px-4 py-3 text-[15px] text-left border border-[#E5E7EB] rounded-[8px] hover:bg-[#F9FAFB] transition-colors duration-150">
            View audit log
          </button>
          <button className="w-full px-4 py-3 text-[15px] text-left border border-[#E5E7EB] rounded-[8px] hover:bg-[#F9FAFB] transition-colors duration-150">
            Data source configuration
          </button>
        </div>
      </Panel>

      <Panel title="Access & Security">
        <div className="space-y-4 text-[15px]">
          <div className="flex justify-between py-2 border-b border-[#E5E7EB]">
            <span className="text-[#6B7280]">User Role</span>
            <span className="text-[#111111]">Administrator</span>
          </div>
          <div className="flex justify-between py-2 border-b border-[#E5E7EB]">
            <span className="text-[#6B7280]">Last Login</span>
            <span className="text-[#111111]">2026-02-19 07:45 UTC</span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-[#6B7280]">Session Expires</span>
            <span className="text-[#111111]">2026-02-19 19:45 UTC</span>
          </div>
        </div>
      </Panel>
    </div>
  );
}
