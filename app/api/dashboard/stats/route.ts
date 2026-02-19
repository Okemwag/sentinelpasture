import { NextResponse } from 'next/server';

export async function GET() {
  const data = {
    activeAlerts: 3,
    regionsMonitored: 47,
    dataSources: 24,
    lastUpdate: '2 minutes ago',
  };

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
