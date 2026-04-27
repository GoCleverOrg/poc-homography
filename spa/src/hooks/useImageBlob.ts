/**
 * Shared hook for loading an image via authenticated fetch, creating a
 * blob object URL, and revoking it on cleanup.
 *
 * Eliminates the duplicated `loadImage` / `loadImageBlob` + cleanup
 * pattern in CameraAnnotatorPage and CameraLineAnnotatorPage.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useTenant } from '../contexts/TenantContext';

interface UseImageBlobReturn {
  /** Current object URL (or empty string if no image loaded). */
  imageUrl: string;
  /** Fetch an image from the given endpoint. Pass `cacheBust` to append a `t` query param. */
  loadImage: (endpoint: string, filename: string, cacheBust?: boolean) => Promise<void>;
  /** Revoke the current blob URL (if any) and reset to empty. */
  clearImage: () => void;
}

export function useImageBlob(): UseImageBlobReturn {
  const { credentials } = useAuth();
  const { selectedTenantId } = useTenant();
  const [imageUrl, setImageUrl] = useState('');
  const urlRef = useRef('');

  // Keep ref in sync for the cleanup function
  useEffect(() => {
    urlRef.current = imageUrl;
  }, [imageUrl]);

  const revokePrev = useCallback((prev: string) => {
    if (prev) URL.revokeObjectURL(prev);
  }, []);

  const clearImage = useCallback(() => {
    setImageUrl((prev) => {
      revokePrev(prev);
      return '';
    });
  }, [revokePrev]);

  const loadImage = useCallback(
    async (endpoint: string, filename: string, cacheBust = false) => {
      if (!credentials || !selectedTenantId || !filename) return;

      const encoded = btoa(`${credentials.username}:${credentials.password}`);
      const params = new URLSearchParams({
        tenant_id: selectedTenantId,
        image_filename: filename,
      });
      if (cacheBust) params.set('t', String(Date.now()));

      try {
        const resp = await fetch(`${endpoint}?${params}`, {
          headers: { Authorization: `Basic ${encoded}` },
        });
        if (!resp.ok) return;
        const blob = await resp.blob();
        const objectUrl = URL.createObjectURL(blob);
        setImageUrl((prev) => {
          revokePrev(prev);
          return objectUrl;
        });
      } catch {
        // Silently ignore network errors
      }
    },
    [credentials, selectedTenantId, revokePrev],
  );

  // Revoke blob URL on unmount
  useEffect(() => {
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  return { imageUrl, loadImage, clearImage };
}
