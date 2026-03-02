// API Client for dashboard backend communication

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

export interface ApiResponse<T> {
  data: T;
  success: boolean;
  error?: string;
  metadata?: {
    modelVersion: string;
    lastUpdated: string;
    confidence: string;
  };
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  // Dashboard Overview
  async getStabilityIndex() {
    return this.request<{
      value: number;
      trend: 'up' | 'down' | 'stable';
      confidence: 'High' | 'Medium' | 'Low';
      change: string;
    }>('/dashboard/stability');
  }

  async getQuickStats() {
    return this.request<{
      activeAlerts: number;
      regionsMonitored: number;
      dataSources: number;
      lastUpdate: string;
    }>('/dashboard/stats');
  }

  async getRiskFactors() {
    return this.request<Array<{
      category: string;
      description: string;
      icon: string;
    }>>('/dashboard/risk-factors');
  }

  // Regional Data
  async getRegionalData() {
    return this.request<Array<{
      region: string;
      population: string;
      stabilityIndex: number;
      trend: string;
      primaryDriver: string;
      confidence: string;
    }>>('/regional/data');
  }

  async getRegionalMap() {
    return this.request<Array<{
      id: string;
      name: string;
      riskLevel: 'low' | 'moderate' | 'high';
      primaryDriver: string;
      secondaryDriver: string;
      confidence: string;
      coordinates: { x: number; y: number; width: number; height: number };
    }>>('/regional/map');
  }

  // Drivers
  async getDrivers() {
    return this.request<Array<{
      label: string;
      percentage: number;
      trend: 'up' | 'down' | 'stable';
      confidence: string;
    }>>('/drivers/list');
  }

  // Interventions
  async getInterventions(region?: string) {
    return this.request<Array<{
      category: string;
      expectedImpact: string;
      timeToEffect: string;
      costBand: string;
      confidence: string;
    }>>(`/interventions/list${region ? `?region=${region}` : ''}`);
  }

  // Outcomes
  async getOutcomes() {
    return this.request<Array<{
      intervention: string;
      deployed: string;
      riskBefore: number;
      riskAfter: number;
      trend: string;
      commentary: string;
    }>>('/outcomes/list');
  }

  async getOutcomesChart() {
    return this.request<Array<{
      date: string;
      value: number;
    }>>('/outcomes/chart');
  }

  // Alerts
  async getAlerts() {
    return this.request<Array<{
      id: number;
      severity: 'elevated' | 'moderate';
      title: string;
      description: string;
      timestamp: string;
      region: string;
      status: 'active' | 'monitoring' | 'resolved';
    }>>('/alerts/list');
  }

  async getAlertStats() {
    return this.request<{
      active: number;
      monitoring: number;
      resolved24h: number;
    }>('/alerts/stats');
  }

  // Reports
  async getReports() {
    return this.request<Array<{
      title: string;
      type: string;
      date: string;
      size: string;
      downloadUrl: string;
    }>>('/reports/list');
  }

  // AI Engine Integration
  async processSignals(signals: any[]) {
    return this.request<{
      assessment: any;
      indicators: any[];
      recommendations: any[];
    }>('/ai/process', {
      method: 'POST',
      body: JSON.stringify({ signals }),
    });
  }

  async getPredictions(region: string, timeframe: string) {
    return this.request<{
      predictions: any[];
      confidence: number;
    }>(`/ai/predict?region=${region}&timeframe=${timeframe}`);
  }
}

export const apiClient = new ApiClient();
