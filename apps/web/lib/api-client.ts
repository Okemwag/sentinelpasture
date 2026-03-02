// API Client for dashboard backend communication

import { authHeader, getApiBaseUrl, getApiRootUrl, type AuthSession, type AuthUser } from "@/lib/auth-session";

const API_BASE_URL = getApiBaseUrl();
const API_ROOT_URL = getApiRootUrl();

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
          ...authHeader(),
          ...options?.headers,
        },
      });

      if (!response.ok) {
        if (response.status === 401 && typeof window !== 'undefined') {
          window.localStorage.removeItem('governance-intel-auth');
        }
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Request failed:', error);
      throw error;
    }
  }

  async login(username: string, password: string): Promise<AuthSession> {
    const response = await fetch(`${API_ROOT_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      throw new Error('Invalid credentials');
    }

    return await response.json();
  }

  async getCurrentUser(): Promise<AuthUser> {
    const response = await fetch(`${API_ROOT_URL}/auth/me`, {
      headers: {
        'Content-Type': 'application/json',
        ...authHeader(),
      },
    });

    if (!response.ok) {
      throw new Error('Unauthorized');
    }

    return await response.json();
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
    }>('/process', {
      method: 'POST',
      body: JSON.stringify({ signals }),
    });
  }

  async getPredictions(region: string, timeframe: string) {
    return this.request<{
      predictions: any[];
      confidence: number;
    }>(`/predict?region=${region}&timeframe=${timeframe}`);
  }
}

export const apiClient = new ApiClient();
