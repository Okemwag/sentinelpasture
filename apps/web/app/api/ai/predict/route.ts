import { NextRequest, NextResponse } from 'next/server';
import { aiEngineService } from '@/lib/ai-engine-service';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const region = searchParams.get('region') || 'national';
    const timeframe = searchParams.get('timeframe') || '7d';

    const result = await aiEngineService.predictRisk(region, timeframe);

    return NextResponse.json({
      data: result,
      success: true,
      metadata: {
        modelVersion: 'v2.4.1',
        lastUpdated: new Date().toISOString(),
        confidence: 'High',
      },
    });
  } catch (error) {
    console.error('AI prediction error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to generate predictions' },
      { status: 500 }
    );
  }
}
