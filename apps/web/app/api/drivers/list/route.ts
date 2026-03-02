import { NextResponse } from 'next/server';

export async function GET() {
  const data = [
    {
      label: 'Economic stress',
      percentage: 34,
      trend: 'up' as const,
      confidence: 'High',
    },
    {
      label: 'Climate anomaly',
      percentage: 28,
      trend: 'up' as const,
      confidence: 'Medium',
    },
    {
      label: 'Mobility disruption',
      percentage: 18,
      trend: 'stable' as const,
      confidence: 'High',
    },
    {
      label: 'Education attendance decline',
      percentage: 12,
      trend: 'down' as const,
      confidence: 'Medium',
    },
    {
      label: 'Incident contagion',
      percentage: 8,
      trend: 'up' as const,
      confidence: 'High',
    },
  ];

  return NextResponse.json({
    data,
    success: true,
    metadata: {
      modelVersion: 'v2.4.1',
      lastUpdated: new Date().toISOString(),
      confidence: 'High',
    },
  });
}
