"use client";

interface DataPoint {
  date: string;
  value: number;
}

interface LineChartProps {
  data: DataPoint[];
  height?: number;
}

export default function LineChart({ data, height = 200 }: LineChartProps) {
  if (data.length === 0) return null;

  const maxValue = Math.max(...data.map((d) => d.value));
  const minValue = Math.min(...data.map((d) => d.value));
  const range = maxValue - minValue;
  const padding = 40;
  const chartWidth = 600;
  const chartHeight = height;

  const points = data
    .map((point, index) => {
      const x = (index / (data.length - 1)) * (chartWidth - padding * 2) + padding;
      const y =
        chartHeight -
        padding -
        ((point.value - minValue) / range) * (chartHeight - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${chartWidth} ${chartHeight}`}
      className="w-full"
      style={{ maxWidth: "100%" }}
    >
      <polyline
        points={points}
        fill="none"
        stroke="#111111"
        strokeWidth="2"
      />
      
      {data.map((point, index) => {
        const x = (index / (data.length - 1)) * (chartWidth - padding * 2) + padding;
        const y =
          chartHeight -
          padding -
          ((point.value - minValue) / range) * (chartHeight - padding * 2);
        return (
          <circle
            key={index}
            cx={x}
            cy={y}
            r="3"
            fill="#111111"
          />
        );
      })}

      <line
        x1={padding}
        y1={chartHeight - padding}
        x2={chartWidth - padding}
        y2={chartHeight - padding}
        stroke="#E5E7EB"
        strokeWidth="1"
      />
      
      <line
        x1={padding}
        y1={padding}
        x2={padding}
        y2={chartHeight - padding}
        stroke="#E5E7EB"
        strokeWidth="1"
      />
    </svg>
  );
}
