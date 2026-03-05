"use client";

import { useEffect, useState } from "react";

import PageHeader from "@/components/dashboard/page-header";
import Panel from "@/components/dashboard/panel";
import DriverBar from "@/components/dashboard/driver-bar";
import { apiClient } from "@/lib/api-client";

type Driver = {
  label: string;
  percentage: number;
  trend: "up" | "down" | "stable";
  confidence: string;
};

export default function DriversPage() {
  const [drivers, setDrivers] = useState<Driver[]>([]);
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
        const response = await apiClient.getDrivers();
        if (!active) {
          return;
        }
        setDrivers(response.data);
        if (response.metadata) {
          setMetadata(response.metadata);
        }
      } catch (loadError) {
        if (!active) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : "Unable to load drivers");
      }
    }

    void load();
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <PageHeader title="Risk Drivers" />

      {error ? (
        <div className="rounded-[8px] border border-[#FECACA] bg-[#FEF2F2] p-4 text-[13px] text-[#991B1B]">
          {error}
        </div>
      ) : null}

      <Panel
        title="Driver Breakdown"
        metadata={{
          modelVersion: metadata.modelVersion,
          lastUpdated: metadata.lastUpdated,
          confidence: metadata.confidence,
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
          {drivers.map((driver) => (
            <div key={driver.label}>
              <div className="mb-1 font-medium text-[#111111]">{driver.label}</div>
              <div className="text-[#6B7280]">
                Current modeled contribution is {driver.percentage.toFixed(1)}% with {driver.confidence.toLowerCase()} confidence.
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
