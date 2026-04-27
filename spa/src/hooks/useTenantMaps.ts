import { useEffect, useState } from 'react';
import { useTenant } from '../contexts/TenantContext';
import { useAuth } from '../contexts/AuthContext';

interface UseTenantMapsResult {
  mapIds: string[] | null;
  loading: boolean;
}

export function useTenantMaps(): UseTenantMapsResult {
  const { selectedTenantId } = useTenant();
  const { credentials } = useAuth();
  const [mapIds, setMapIds] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!selectedTenantId || !credentials) {
      setMapIds(null);
      return;
    }

    let cancelled = false;
    setLoading(true);

    const encoded = btoa(`${credentials.username}:${credentials.password}`);
    fetch(`/gcp/api/map-ids/?tenant_id=${encodeURIComponent(selectedTenantId)}`, {
      headers: { Authorization: `Basic ${encoded}` },
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Failed to load map IDs (${res.status})`);
        return res.json() as Promise<{ map_ids: string[] }>;
      })
      .then(({ map_ids }) => {
        if (!cancelled) setMapIds(map_ids);
      })
      .catch(() => {
        if (!cancelled) setMapIds(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedTenantId, credentials]);

  return { mapIds, loading };
}
