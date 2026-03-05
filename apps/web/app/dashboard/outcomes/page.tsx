"use client";

import { useEffect, useState } from "react";

import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import LineChart from "@/components/dashboard/line-chart";
import { apiClient } from "@/lib/api-client";

type Outcome = {
  intervention: string;
  deployed: string;
  riskBefore: number;
  riskAfter: number;
  trend: string;
  commentary: string;
};

type ChartPoint = {
  date: string;
  value: number;
};

export default function OutcomesPage() {
  const [rows, setRows] = useState<Outcome[]>([]);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [metadata, setMetadata] = useState({
    modelVersion: "loading",
    lastUpdated: "Loading...",
    confidence: "Medium",
  });
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const [outcomesResponse, chartResponse] = await Promise.all([
          apiClient.getOutcomes(),
          apiClient.getOutcomesChart(),
        ]);
        if (!active) {
          return;
        }
        setRows(outcomesResponse.data);
        setChartData(chartResponse.data);
        if (outcomesResponse.metadata) {
          setMetadata(outcomesResponse.metadata);
        } else if (chartResponse.metadata) {
          setMetadata(chartResponse.metadata);
        }
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load outcomes");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader title="Intervention Outcomes" />

      {error ? (
        <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] p-4 text-[13px] text-[#991B1B]">
          {error}
        </div>
      ) : null}

      <Panel title="Risk Trend Following Intervention">
        <LineChart data={chartData} height={240} />
      </Panel>

      <Panel
        title="Deployed Interventions"
        metadata={{
          modelVersion: metadata.modelVersion,
          lastUpdated: metadata.lastUpdated,
          confidence: metadata.confidence,
        }}
      >
        <div className="space-y-6">
          {rows.map((outcome) => (
            <div
              key={`${outcome.intervention}-${outcome.deployed}`}
              className="border-b border-[#E5E7EB] pb-6 last:border-0 last:pb-0"
            >
              <div className="mb-4 grid grid-cols-2 gap-6">
                <div>
                  <div className="mb-1 text-[13px] text-[#6B7280]">Intervention</div>
                  <div className="text-[15px] font-medium text-[#111111]">{outcome.intervention}</div>
                </div>
                <div>
                  <div className="mb-1 text-[13px] text-[#6B7280]">Date Deployed</div>
                  <div className="text-[15px] text-[#111111]">{outcome.deployed}</div>
                </div>
              </div>

              <div className="mb-4 grid grid-cols-4 gap-6">
                <div>
                  <div className="mb-1 text-[13px] text-[#6B7280]">Risk Before</div>
                  <div className="text-[18px] font-medium text-[#111111]">{outcome.riskBefore}</div>
                </div>
                <div>
                  <div className="mb-1 text-[13px] text-[#6B7280]">Risk After</div>
                  <div className="text-[18px] font-medium text-[#111111]">{outcome.riskAfter}</div>
                </div>
                <div>
                  <div className="mb-1 text-[13px] text-[#6B7280]">Observed Trend</div>
                  <div className="text-[15px] text-[#111111]">{outcome.trend}</div>
                </div>
                <div>
                  <div className="mb-1 text-[13px] text-[#6B7280]">Change</div>
                  <div className="text-[18px] font-medium text-[#111111]">
                    {outcome.riskAfter - outcome.riskBefore}
                  </div>
                </div>
              </div>

              <div>
                <div className="mb-1 text-[13px] text-[#6B7280]">Commentary</div>
                <div className="text-[15px] leading-relaxed text-[#111111]">{outcome.commentary}</div>
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
