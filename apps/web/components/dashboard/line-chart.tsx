"use client";

interface DataPoint {
  date: string;
  value: number;
}

interface LineChartProps {
  data: DataPoint[];
  height?: number;
  fillColor?: string;
  strokeColor?: string;
  showLabels?: boolean;
}

export default function LineChart({
  data,
  height = 200,
  fillColor = "#C7A56B",
  strokeColor = "#8C6A3D",
  showLabels = true,
}: LineChartProps) {
  if (data.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-[13px] text-[#9CA3AF]"
        style={{ height }}
      >
        No data available
      </div>
    );
  }
  if (data.length === 1) {
    return (
      <div
        className="flex items-center justify-center text-[13px] text-[#9CA3AF]"
        style={{ height }}
      >
        Insufficient data points
      </div>
    );
  }

  const maxValue = Math.max(...data.map((d) => d.value));
  const minValue = Math.min(...data.map((d) => d.value));
  const range = maxValue - minValue || 1;
  const padLeft = showLabels ? 36 : 16;
  const padRight = 16;
  const padTop = 12;
  const padBottom = showLabels ? 32 : 16;
  const chartWidth = 600;
  const chartHeight = height;
  const innerW = chartWidth - padLeft - padRight;
  const innerH = chartHeight - padTop - padBottom;

  const toPoint = (index: number, value: number) => {
    const x = padLeft + (index / (data.length - 1)) * innerW;
    const y = padTop + (1 - (value - minValue) / range) * innerH;
    return { x, y };
  };

  const points = data.map((d, i) => toPoint(i, d.value));
  const polyline = points.map((p) => `${p.x},${p.y}`).join(" ");

  // closed polygon for fill area
  const fillPath = [
    `M ${points[0].x} ${chartHeight - padBottom}`,
    ...points.map((p) => `L ${p.x} ${p.y}`),
    `L ${points[points.length - 1].x} ${chartHeight - padBottom}`,
    "Z",
  ].join(" ");

  // Y-axis labels
  const ySteps = 4;
  const yLabels = Array.from({ length: ySteps + 1 }, (_, i) => {
    const val = minValue + (range / ySteps) * i;
    const y = padTop + (1 - (val - minValue) / range) * innerH;
    return { val: Math.round(val), y };
  });

  // X-axis labels (show first, middle, last)
  const xLabelIndices =
    data.length <= 6
      ? data.map((_, i) => i)
      : [0, Math.floor((data.length - 1) / 2), data.length - 1];

  const lastPoint = points[points.length - 1];

  return (
    <svg
      viewBox={`0 0 ${chartWidth} ${chartHeight}`}
      className="w-full"
      style={{ maxWidth: "100%" }}
    >
      {/* Grid lines */}
      {yLabels.map(({ y, val }) => (
        <g key={val}>
          <line
            x1={padLeft}
            y1={y}
            x2={chartWidth - padRight}
            y2={y}
            stroke="#F3F4F6"
            strokeWidth={1}
            strokeDasharray="4,4"
          />
          {showLabels && (
            <text
              x={padLeft - 6}
              y={y + 4}
              textAnchor="end"
              fontSize={9}
              fill="#9CA3AF"
            >
              {val}
            </text>
          )}
        </g>
      ))}

      {/* Axes */}
      <line
        x1={padLeft}
        y1={padTop}
        x2={padLeft}
        y2={chartHeight - padBottom}
        stroke="#E5E7EB"
        strokeWidth={1}
      />
      <line
        x1={padLeft}
        y1={chartHeight - padBottom}
        x2={chartWidth - padRight}
        y2={chartHeight - padBottom}
        stroke="#E5E7EB"
        strokeWidth={1}
      />

      {/* Fill area */}
      <path d={fillPath} fill={fillColor} fillOpacity={0.12} />

      {/* Line */}
      <polyline
        points={polyline}
        fill="none"
        stroke={strokeColor}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {/* X labels */}
      {showLabels &&
        xLabelIndices.map((i) => {
          const d = data[i];
          const p = points[i];
          const label = new Date(d.date).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          });
          return (
            <text
              key={i}
              x={p.x}
              y={chartHeight - padBottom + 14}
              textAnchor="middle"
              fontSize={9}
              fill="#9CA3AF"
            >
              {label}
            </text>
          );
        })}

      {/* Moving dot on latest point */}
      <circle cx={lastPoint.x} cy={lastPoint.y} r={5} fill={strokeColor} />
      <circle cx={lastPoint.x} cy={lastPoint.y} r={9} fill={strokeColor} fillOpacity={0.15} />
    </svg>
  );
}
