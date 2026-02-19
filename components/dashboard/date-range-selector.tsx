"use client";

import { useState } from "react";

const periods = [
  { label: "Last 7 days", value: "7d" },
  { label: "Last 30 days", value: "30d" },
  { label: "Last 90 days", value: "90d" },
];

export default function DateRangeSelector() {
  const [selected, setSelected] = useState("30d");

  return (
    <div className="flex gap-2">
      {periods.map((period) => (
        <button
          key={period.value}
          onClick={() => setSelected(period.value)}
          className={`
            px-4 py-2 text-[13px] font-normal rounded-[8px]
            transition-colors duration-150
            ${
              selected === period.value
                ? "bg-[#374151] text-white"
                : "bg-white text-[#6B7280] border border-[#E5E7EB] hover:bg-[#F9FAFB]"
            }
          `}
        >
          {period.label}
        </button>
      ))}
    </div>
  );
}
