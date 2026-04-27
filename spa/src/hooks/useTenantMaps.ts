import { useEffect, useState } from 'react';
import { useTenant } from '../contexts/TenantContext';
import client from '../api/client';

interface UseTenantMapsResult {
  mapIds: string[] | null;
  loading: boolean;
}

export function useTenantMaps(): UseTenantMapsResult {
  const { selectedTenantId } = useTenant();
  const [mapIds, setMapIds] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedTenantId) {
      setMapIds(null);
      return;
    }

    const controller = new AbortController();
    setLoading(true);

    client
      .GET('/gcp/api/map-ids/', {
        params: { query: { tenant_id: selectedTenantId } },
        signal: controller.signal,
      })
      .then(({ data, error }) => {
        if (controller.signal.aborted) return;
        if (error) {
          console.error('Failed to load map IDs:', error);
          setMapIds(null);
        } else if (data) {
          setMapIds((data as { map_ids: string[] }).map_ids);
        }
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          console.error('Failed to load map IDs:', err);
          setMapIds(null);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => {
      controller.abort();
    };
  }, [selectedTenantId]);

  return { mapIds, loading };
}
