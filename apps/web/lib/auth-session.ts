export interface AuthUser {
  id: number;
  username: string;
  email: string;
  full_name: string;
  role: string;
}

export interface AuthSession {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
}

const AUTH_STORAGE_KEY = "governance-intel-auth";

export function getApiRootUrl() {
  const configured = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
  return configured.endsWith("/api") ? configured.slice(0, -4) : configured;
}

export function getApiBaseUrl() {
  return `${getApiRootUrl()}/api`;
}

export function storeAuthSession(session: AuthSession) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
}

export function readAuthSession(): AuthSession | null {
  if (typeof window === "undefined") {
    return null;
  }
  const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw) as AuthSession;
  } catch {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return null;
  }
}

export function clearAuthSession() {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
}

export function authHeader() {
  const session = readAuthSession();
  return session ? { Authorization: `Bearer ${session.access_token}` } : {};
}
