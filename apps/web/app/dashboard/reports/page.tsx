"use client";

import { useEffect, useState } from "react";

import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import { apiClient } from "@/lib/api-client";
import { authHeader } from "@/lib/auth-session";

type ReportRow = {
  title: string;
  type: string;
  date: string;
  size: string;
  downloadUrl: string;
};

export default function ReportsPage() {
  const [reports, setReports] = useState<ReportRow[]>([]);
  const [error, setError] = useState("");
  const [downloading, setDownloading] = useState<string>("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const response = await apiClient.getReports();
        if (!active) {
          return;
        }
        setReports(response.data);
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load reports");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  async function handleDownload(report: ReportRow) {
    setError("");
    setDownloading(report.downloadUrl);
    try {
      const response = await fetch(report.downloadUrl, {
        headers: {
          ...authHeader(),
        },
      });
      if (!response.ok) {
        throw new Error(`Download failed: ${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `${report.title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(objectUrl);
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : "Unable to download report");
    } finally {
      setDownloading("");
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Reports & Exports" />

      {error ? (
        <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] p-4 text-[13px] text-[#991B1B]">
          {error}
        </div>
      ) : null}

      <Panel title="Available Reports">
        <div className="space-y-3">
          {reports.map((report) => (
            <div
              key={`${report.title}-${report.date}`}
              className="flex items-center justify-between rounded-[8px] border border-[#E5E7EB] p-4 transition-colors duration-150 hover:bg-[#F9FAFB]"
            >
              <div>
                <div className="mb-1 text-[15px] font-medium text-[#111111]">{report.title}</div>
                <div className="text-[13px] text-[#6B7280]">
                  {report.type} • {report.date} • {report.size}
                </div>
              </div>
              <button
                type="button"
                onClick={() => void handleDownload(report)}
                disabled={downloading === report.downloadUrl}
                className="rounded-[8px] border border-[#E5E7EB] px-4 py-2 text-[13px] text-[#374151] transition-colors duration-150 hover:bg-[#F3F4F6]"
              >
                {downloading === report.downloadUrl ? "Downloading..." : "Download"}
              </button>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="Generate New Report">
        <div className="text-[14px] text-[#6B7280]">
          Report generation workflow is connected to backend list/download capabilities. Scheduled generation API is pending.
        </div>
      </Panel>
    </div>
  );
}
