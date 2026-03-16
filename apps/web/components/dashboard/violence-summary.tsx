import LineChart from "@/components/dashboard/line-chart";

type ViolencePoint = {
  date: string;
  totalEvents: number;
  totalFatalities: number;
  demonstrationsEvents: number;
  civilianTargetingEvents: number;
  politicalViolenceEvents: number;
};

export default function ViolenceSummary({ series }: { series: ViolencePoint[] }) {
  if (!series.length) {
    return null;
  }

  const latest = series[series.length - 1];

  return (
    <div className="bg-white border border-[#E5E7EB] rounded-[8px] p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-[15px] font-semibold text-[#111111]">
            Conflict & Violence Signals — National
          </h2>
          <p className="text-[12px] text-[#6B7280] mt-0.5">
            Monthly demonstrations and political violence events informing the baseline risk score.
          </p>
        </div>
        <div className="text-[11px] text-[#9CA3AF]">
          Latest period: {latest.date}
        </div>
      </div>

      <LineChart
        data={series.map((row) => ({
          date: row.date,
          value: row.totalEvents,
        }))}
        height={180}
      />

      <div className="mt-4 grid grid-cols-2 gap-3 text-[12px] text-[#6B7280]">
        <Metric
          label="Demonstrations"
          value={latest.demonstrationsEvents}
          color="#4A7490"
        />
        <Metric
          label="Civilian targeting events"
          value={latest.civilianTargetingEvents}
          color="#8C6A3D"
        />
        <Metric
          label="Political violence events"
          value={latest.politicalViolenceEvents}
          color="#B0413E"
        />
        <Metric
          label="Total fatalities (all types)"
          value={latest.totalFatalities}
          color="#7A9B70"
        />
      </div>
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="flex items-center gap-2">
        <span
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span>{label}</span>
      </span>
      <span className="font-semibold text-[#111111]">{value}</span>
    </div>
  );
}

