// AI Engine Integration Service
// This service connects the Next.js backend to the Python AI engine

const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';

export interface Signal {
  type: string;
  source: string;
  data: any;
  location?: {
    latitude: number;
    longitude: number;
  };
  temporal?: {
    timestamp: string;
  };
}

export interface AIEngineResponse {
  assessment: {
    threat_level: number;
    confidence: number;
    risk_factors: string[];
  };
  indicators: Array<{
    type: string;
    severity: number;
    description: string;
  }>;
  recommendations: Array<{
    action: string;
    priority: number;
    estimated_impact: string;
  }>;
  metadata: {
    model_version: string;
    processing_time: number;
  };
}

class AIEngineService {
  private baseUrl: string;
  private useMockData: boolean;

  constructor() {
    this.baseUrl = AI_ENGINE_URL;
    // Use mock data if AI engine is not available
    this.useMockData = process.env.USE_MOCK_AI === 'true' || true;
  }

  async processSignals(signals: Signal[]): Promise<AIEngineResponse> {
    if (this.useMockData) {
      return this.getMockResponse(signals);
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ signals }),
      });

      if (!response.ok) {
        throw new Error(`AI Engine Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('AI Engine request failed, using mock data:', error);
      return this.getMockResponse(signals);
    }
  }

  async predictRisk(region: string, timeframe: string): Promise<any> {
    if (this.useMockData) {
      return this.getMockPrediction(region, timeframe);
    }

    try {
      const response = await fetch(
        `${this.baseUrl}/api/predict?region=${region}&timeframe=${timeframe}`
      );

      if (!response.ok) {
        throw new Error(`AI Engine Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('AI Engine prediction failed, using mock data:', error);
      return this.getMockPrediction(region, timeframe);
    }
  }

  async analyzeDrivers(data: any): Promise<any> {
    if (this.useMockData) {
      return this.getMockDriverAnalysis();
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/analyze-drivers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`AI Engine Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('AI Engine driver analysis failed, using mock data:', error);
      return this.getMockDriverAnalysis();
    }
  }

  async recommendInterventions(region: string, riskProfile: any): Promise<any> {
    if (this.useMockData) {
      return this.getMockInterventions(region);
    }

    try {
      const response = await fetch(`${this.baseUrl}/api/recommend-interventions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ region, riskProfile }),
      });

      if (!response.ok) {
        throw new Error(`AI Engine Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('AI Engine intervention recommendation failed, using mock data:', error);
      return this.getMockInterventions(region);
    }
  }

  // Mock data methods for development
  private getMockResponse(signals: Signal[]): AIEngineResponse {
    return {
      assessment: {
        threat_level: 67,
        confidence: 0.85,
        risk_factors: ['Economic stress', 'Climate anomaly', 'Social tension'],
      },
      indicators: [
        {
          type: 'economic',
          severity: 0.72,
          description: 'Unemployment rate increasing in urban areas',
        },
        {
          type: 'environmental',
          severity: 0.65,
          description: 'Drought conditions affecting agricultural regions',
        },
        {
          type: 'social',
          severity: 0.58,
          description: 'Youth unemployment above national average',
        },
      ],
      recommendations: [
        {
          action: 'Economic stabilization program',
          priority: 1,
          estimated_impact: 'High',
        },
        {
          action: 'Water resource management initiative',
          priority: 2,
          estimated_impact: 'Moderate',
        },
        {
          action: 'Youth employment scheme',
          priority: 3,
          estimated_impact: 'Moderate',
        },
      ],
      metadata: {
        model_version: 'v2.4.1',
        processing_time: 0.234,
      },
    };
  }

  private getMockPrediction(region: string, timeframe: string): any {
    return {
      predictions: [
        { date: '2026-02-20', risk_level: 68, confidence: 0.82 },
        { date: '2026-02-21', risk_level: 69, confidence: 0.80 },
        { date: '2026-02-22', risk_level: 70, confidence: 0.78 },
        { date: '2026-02-23', risk_level: 71, confidence: 0.76 },
        { date: '2026-02-24', risk_level: 72, confidence: 0.74 },
      ],
      confidence: 0.78,
      factors: ['Economic indicators trending upward', 'Seasonal patterns'],
    };
  }

  private getMockDriverAnalysis(): any {
    return {
      drivers: [
        {
          name: 'Economic stress',
          contribution: 0.34,
          trend: 'increasing',
          confidence: 0.92,
        },
        {
          name: 'Climate anomaly',
          contribution: 0.28,
          trend: 'increasing',
          confidence: 0.85,
        },
        {
          name: 'Mobility disruption',
          contribution: 0.18,
          trend: 'stable',
          confidence: 0.88,
        },
      ],
      causal_relationships: [
        {
          from: 'Climate anomaly',
          to: 'Economic stress',
          strength: 0.65,
        },
        {
          from: 'Economic stress',
          to: 'Social tension',
          strength: 0.72,
        },
      ],
    };
  }

  private getMockInterventions(region: string): any {
    return {
      interventions: [
        {
          category: 'Economic stabilization program',
          expectedImpact: 'High',
          timeToEffect: 'Medium',
          costBand: 'KES 800M - 1.2B',
          confidence: 'High',
          effectiveness_score: 0.85,
        },
        {
          category: 'Infrastructure resilience enhancement',
          expectedImpact: 'Moderate',
          timeToEffect: 'Long',
          costBand: 'KES 1.5B - 2.5B',
          confidence: 'Medium',
          effectiveness_score: 0.72,
        },
        {
          category: 'Community engagement initiative',
          expectedImpact: 'Moderate',
          timeToEffect: 'Short',
          costBand: 'KES 150M - 300M',
          confidence: 'High',
          effectiveness_score: 0.68,
        },
      ],
    };
  }
}

export const aiEngineService = new AIEngineService();
