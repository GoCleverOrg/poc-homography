import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { setClientCredentials, setOnUnauthorized } from '../api/client';

interface Credentials {
  username: string;
  password: string;
}

interface AuthContextValue {
  credentials: Credentials | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [credentials, setCredentials] = useState<Credentials | null>(null);

  const logout = useCallback(() => {
    setCredentials(null);
    setClientCredentials(null);
  }, []);

  // Wire the 401 handler once on mount.
  useEffect(() => {
    setOnUnauthorized(logout);
  }, [logout]);

  // Keep the API client in sync whenever credentials change.
  useEffect(() => {
    setClientCredentials(credentials);
  }, [credentials]);

  const login = useCallback(async (username: string, password: string) => {
    const encoded = btoa(`${username}:${password}`);
    const res = await fetch('/health', {
      headers: { Authorization: `Basic ${encoded}` },
    });

    if (!res.ok) {
      throw new Error(
        res.status === 401
          ? 'Invalid username or password'
          : `Health check failed (${res.status})`,
      );
    }

    const creds: Credentials = { username, password };
    setCredentials(creds);
    setClientCredentials(creds);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ credentials, isAuthenticated: credentials !== null, login, logout }),
    [credentials, login, logout],
  );

  return <AuthContext value={value}>{children}</AuthContext>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return ctx;
}
