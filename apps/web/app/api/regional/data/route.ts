import { NextResponse } from 'next/server';

export async function GET() {
  const data = [
    {
      region: 'Nairobi County',
      population: '4.4M',
      stabilityIndex: 72,
      trend: 'Stable',
      primaryDriver: 'Economic stress',
      confidence: 'High',
    },
    {
      region: 'Mombasa County',
      population: '1.2M',
      stabilityIndex: 68,
      trend: 'Moderating',
      primaryDriver: 'Infrastructure strain',
      confidence: 'High',
    },
    {
      region: 'Kisumu County',
      population: '1.1M',
      stabilityIndex: 81,
      trend: 'Stable',
      primaryDriver: 'Mobility disruption',
      confidence: 'Medium',
    },
    {
      region: 'Nakuru County',
      population: '2.2M',
      stabilityIndex: 75,
      trend: 'Stable',
      primaryDriver: 'Climate anomaly',
      confidence: 'High',
    },
    {
      region: 'Turkana County',
      population: '926K',
      stabilityIndex: 64,
      trend: 'Elevated',
      primaryDriver: 'Climate anomaly',
      confidence: 'Medium',
    },
    {
      region: 'Kiambu County',
      population: '2.4M',
      stabilityIndex: 78,
      trend: 'Stable',
      primaryDriver: 'Economic stress',
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
