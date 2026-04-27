import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { useAuth } from './AuthContext';
import { setClientTenantId } from '../api/client';

const STORAGE_KEY = 'smartterminal_tenant_id';

export interface Tenant {
  id: string;
  name: string;
  description?: string;
}

interface TenantContextValue {
  tenants: Tenant[] | null;
  selectedTenantId: string | null;
  setSelectedTenantId: (id: string) => void;
  loading: boolean;
  error: string | null;
}

const TenantContext = createContext<TenantContextValue | null>(null);

export function TenantProvider({ children }: { children: ReactNode }) {
  const { isAuthenticated, credentials } = useAuth();
  const [tenants, setTenants] = useState<Tenant[] | null>(null);
  const [selectedTenantId, setSelectedTenantIdState] = useState<string | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const setSelectedTenantId = useCallback((id: string) => {
    setSelectedTenantIdState(id);
    localStorage.setItem(STORAGE_KEY, id);
  }, []);

  // Keep the API client in sync whenever selectedTenantId changes.
  useEffect(() => {
    setClientTenantId(selectedTenantId);
  }, [selectedTenantId]);

  // Fetch tenant list when authenticated.
  useEffect(() => {
    if (!isAuthenticated || !credentials) {
      setTenants(null);
      setSelectedTenantIdState(null);
      setClientTenantId(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    const encoded = btoa(`${credentials.username}:${credentials.password}`);
    fetch('/gcp/api/tenants/', {
      headers: { Authorization: `Basic ${encoded}` },
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Failed to load tenants (${res.status})`);
        return res.json() as Promise<{ tenants: Tenant[] }>;
      })
      .then(({ tenants: list }) => {
        if (cancelled) return;
        setTenants(list);

        // Restore from localStorage if the tenant still exists.
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored && list.some((t) => t.id === stored)) {
          setSelectedTenantIdState(stored);
          setClientTenantId(stored);
        } else if (list.length > 0) {
          // Auto-select first tenant.
          setSelectedTenantIdState(list[0].id);
          setClientTenantId(list[0].id);
          localStorage.setItem(STORAGE_KEY, list[0].id);
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Unknown error');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [isAuthenticated, credentials]);

  const value = useMemo<TenantContextValue>(
    () => ({
      tenants,
      selectedTenantId,
      setSelectedTenantId,
      loading,
      error,
    }),
    [tenants, selectedTenantId, setSelectedTenantId, loading, error],
  );

  return <TenantContext value={value}>{children}</TenantContext>;
}

export function useTenant(): TenantContextValue {
  const ctx = useContext(TenantContext);
  if (!ctx) {
    throw new Error('useTenant must be used within a TenantProvider');
  }
  return ctx;
}
