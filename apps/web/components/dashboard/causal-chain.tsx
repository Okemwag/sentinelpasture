"use client";

interface ChainNode {
    id: string;
    label: string;
    sublabel?: string;
}

interface CausalChainProps {
    nodes?: ChainNode[];
}

const DEFAULT_NODES: ChainNode[] = [
    { id: "rainfall", label: "Rainfall Anomaly", sublabel: "+210% above norm" },
    { id: "livestock", label: "Livestock Migration", sublabel: "Herd displacement" },
    { id: "market", label: "Market Disruption", sublabel: "Supply chain stress" },
    { id: "instability", label: "Instability Pressure", sublabel: "Security & social" },
];

export default function CausalChain({ nodes = DEFAULT_NODES }: CausalChainProps) {
    const nodeW = 150;
    const nodeH = 56;
    const arrowGap = 32;
    const totalW = nodes.length * nodeW + (nodes.length - 1) * arrowGap;
    const svgH = nodeH + 32;

    return (
        <div className="overflow-x-auto">
            <svg
                viewBox={`0 0 ${totalW} ${svgH}`}
                className="w-full"
                style={{ minWidth: Math.min(totalW, 300), maxWidth: "100%" }}
            >
                <defs>
                    <marker
                        id="arrow"
                        markerWidth="8"
                        markerHeight="8"
                        refX="6"
                        refY="4"
                        orient="auto"
                    >
                        <path d="M1,1 L7,4 L1,7 Z" fill="#9CA3AF" />
                    </marker>
                </defs>

                {nodes.map((node, idx) => {
                    const x = idx * (nodeW + arrowGap);
                    const cx = x + nodeW / 2;

                    // Draw arrow between nodes
                    const hasArrow = idx < nodes.length - 1;
                    const arrowX1 = x + nodeW;
                    const arrowX2 = x + nodeW + arrowGap - 2;
                    const arrowY = nodeH / 2 + 16;

                    return (
                        <g key={node.id}>
                            {/* Node box */}
                            <rect
                                x={x}
                                y={16}
                                width={nodeW}
                                height={nodeH}
                                rx={6}
                                fill="#F9FAFB"
                                stroke="#D1D5DB"
                                strokeWidth={1.5}
                            />
                            <text
                                x={cx}
                                y={16 + nodeH / 2 - 7}
                                textAnchor="middle"
                                className="text-[11px] font-semibold"
                                fill="#111111"
                                fontSize={11}
                                fontWeight={600}
                            >
                                {node.label}
                            </text>
                            {node.sublabel && (
                                <text
                                    x={cx}
                                    y={16 + nodeH / 2 + 10}
                                    textAnchor="middle"
                                    fill="#6B7280"
                                    fontSize={10}
                                >
                                    {node.sublabel}
                                </text>
                            )}

                            {/* Arrow */}
                            {hasArrow && (
                                <line
                                    x1={arrowX1}
                                    y1={arrowY}
                                    x2={arrowX2}
                                    y2={arrowY}
                                    stroke="#9CA3AF"
                                    strokeWidth={1.5}
                                    markerEnd="url(#arrow)"
                                />
                            )}
                        </g>
                    );
                })}
            </svg>
        </div>
    );
}
