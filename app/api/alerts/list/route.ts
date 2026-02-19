import { NextResponse } from 'next/server';

export async function GET() {
  const data = [
    {
      id: 1,
      severity: 'elevated' as const,
      title: 'Economic stress indicators rising in Turkana County',
      description: 'Unemployment rate increased by 2.3% over the past 30 days. Food prices up 8.4%.',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      region: 'Turkana County',
      status: 'active' as const,
    },
    {
      id: 2,
      severity: 'moderate' as const,
      title: 'Drought conditions persisting in Northern regions',
      description: 'Rainfall 22% below seasonal average. Water scarcity affecting 3 counties.',
      timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      region: 'Northern Region',
      status: 'active' as const,
    },
    {
      id: 3,
      severity: 'elevated' as const,
      title: 'Youth unemployment spike in Nairobi County',
      description: 'Youth unemployment reached 14.2%, up from 12.1% last quarter.',
      timestamp: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
      region: 'Nairobi County',
      status: 'monitoring' as const,
    },
    {
      id: 4,
      severity: 'moderate' as const,
      title: 'Healthcare capacity strain in Mombasa',
      description: 'Hospital bed occupancy at 89%. Staffing levels below recommended threshold.',
      timestamp: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
      region: 'Mombasa County',
      status: 'resolved' as const,
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
