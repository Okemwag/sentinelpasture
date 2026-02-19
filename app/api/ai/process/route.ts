import { NextRequest, NextResponse } from 'next/server';
import { aiEngineService } from '@/lib/ai-engine-service';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { signals } = body;

    if (!signals || !Array.isArray(signals)) {
      return NextResponse.json(
        { success: false, error: 'Invalid signals data' },
        { status: 400 }
      );
    }

    const result = await aiEngineService.processSignals(signals);

    return NextResponse.json({
      data: result,
      success: true,
      metadata: {
        modelVersion: result.metadata.model_version,
        lastUpdated: new Date().toISOString(),
        confidence: 'High',
        processingTime: result.metadata.processing_time,
      },
    });
  } catch (error) {
    console.error('AI processing error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to process signals' },
      { status: 500 }
    );
  }
}
