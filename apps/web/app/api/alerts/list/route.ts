import { proxyBackendGet } from '@/lib/backend-proxy';

export async function GET() {
  return proxyBackendGet('/api/alerts/list');
}
