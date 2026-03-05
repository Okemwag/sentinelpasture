"use client";

import { useEffect, useState } from "react";

import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import { apiClient } from "@/lib/api-client";

type Intervention = {
  category: string;
  expectedImpact: string;
  timeToEffect: string;
  costBand: string;
  confidence: string;
};

export default function InterventionsPage() {
  const [rows, setRows] = useState<Intervention[]>([]);
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
        const response = await apiClient.getInterventions();
        if (!active) {
          return;
        }
        setRows(response.data);
        if (response.metadata) {
          setMetadata(response.metadata);
        }
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load interventions");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader title="Intervention Analysis" />

      {error ? (
        <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] p-4 text-[13px] text-[#991B1B]">
          {error}
        </div>
      ) : null}

      <div className="rounded-[8px] border border-[#E5E7EB] bg-[#F9FAFB] p-4">
        <p className="text-[13px] leading-relaxed text-[#6B7280]">
          Recommendations are generated from the highest current modeled pressure zone unless a region filter is supplied.
          Final decision authority remains with designated governance bodies.
        </p>
      </div>

      <Panel
        title="Ranked Intervention Options"
        metadata={{
          modelVersion: metadata.modelVersion,
          lastUpdated: metadata.lastUpdated,
          confidence: metadata.confidence,
          showExplainLink: true,
        }}
      >
        <div className="overflow-x-auto -mx-6 px-6">
          <table className="w-full min-w-[640px]">
            <thead>
              <tr className="border-b border-[#E5E7EB]">
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Intervention Category</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Expected Impact</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Time to Effect</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Cost Band</th>
                <th className="px-4 py-3 text-left text-[13px] font-medium text-[#6B7280]">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((intervention) => (
                <tr
                  key={intervention.category}
                  className="border-b border-[#E5E7EB] last:border-0 transition-colors duration-150 hover:bg-[#F9FAFB]"
                >
                  <td className="px-4 py-3 text-[15px] text-[#111111]">{intervention.category}</td>
                  <td className="px-4 py-3 text-[15px] text-[#111111]">{intervention.expectedImpact}</td>
                  <td className="px-4 py-3 text-[15px] text-[#6B7280]">{intervention.timeToEffect}</td>
                  <td className="whitespace-nowrap px-4 py-3 text-[15px] text-[#6B7280]">{intervention.costBand}</td>
                  <td className="px-4 py-3 text-[15px] text-[#111111]">{intervention.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
