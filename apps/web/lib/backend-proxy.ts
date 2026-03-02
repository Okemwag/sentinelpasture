import { NextResponse } from 'next/server';

const BACKEND_API_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

export async function proxyBackendGet(path: string) {
  try {
    const response = await fetch(`${BACKEND_API_URL}${path}`, {
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.statusText}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error(`Backend proxy failed for ${path}:`, error);
    return NextResponse.json(
      { success: false, error: 'Failed to reach backend API' },
      { status: 502 }
    );
  }
}
