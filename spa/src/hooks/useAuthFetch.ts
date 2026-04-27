/**
 * Shared authenticated-fetch hooks used by pages that call REST endpoints
 * outside of the openapi-fetch `client` (e.g. endpoints that return an
 * envelope `{ status: 'success', data: T }`).
 *
 * Eliminates the duplicated `authHeaders` / `apiFetch` / `apiJson` pattern
 * that was copy-pasted across CameraSurveyPage and CameraEvaluationPage.
 */

import { useCallback, useMemo } from 'react';
import { useAuth } from '../contexts/AuthContext';

/**
 * Returns a stable `Record<string, string>` containing the Basic-auth header
 * (or empty when not authenticated). Re-derived only when credentials change.
 */
export function useAuthHeaders(): Record<string, string> {
  const { credentials } = useAuth();
  return useMemo((): Record<string, string> => {
    if (!credentials) return {};
    const encoded = btoa(`${credentials.username}:${credentials.password}`);
    return { Authorization: `Basic ${encoded}` };
  }, [credentials]);
}

/**
 * Returns a `fetch`-like helper that automatically injects auth headers.
 * Drop-in replacement for the per-page `apiFetch` callbacks.
 */
export function useApiFetch(): (url: string, init?: RequestInit) => Promise<Response> {
  const headers = useAuthHeaders();
  return useCallback(
    (url: string, init?: RequestInit) =>
      fetch(url, {
        ...init,
        headers: { ...headers, ...init?.headers },
      }),
    [headers],
  );
}

/**
 * Returns an async helper that calls `apiFetch`, reads the JSON envelope,
 * and throws on non-success. Drop-in replacement for the per-page `apiJson`
 * callbacks.
 */
export function useApiJson(): <T>(url: string, init?: RequestInit) => Promise<T> {
  const apiFetch = useApiFetch();
  return useCallback(
    async <T,>(url: string, init?: RequestInit): Promise<T> => {
      const res = await apiFetch(url, init);
      const json = (await res.json()) as {
        status: string;
        data: T;
        message?: string;
      };
      if (json.status !== 'success')
        throw new Error(json.message ?? 'Request failed');
      return json.data;
    },
    [apiFetch],
  );
}
