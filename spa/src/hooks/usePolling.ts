/**
 * Shared hook for interval-based status polling with automatic cleanup.
 *
 * Eliminates the duplicated `pollRef` + `pollStatusRef` + cleanup pattern
 * in CameraSurveyPage, CameraEvaluationPage, and StressTestTab.
 */

import { useCallback, useEffect, useRef } from 'react';

interface UsePollingReturn {
  /** Start polling a given `targetId` at `intervalMs` (default 1000). */
  startPolling: (targetId: string, intervalMs?: number) => void;
  /** Stop the current polling interval. */
  stopPolling: () => void;
}

/**
 * @param pollFn - The async function to call on each tick. Receives the
 *   `targetId` passed to `startPolling`. The function is captured by ref
 *   so it can close over the latest state without causing restarts.
 */
export function usePolling(
  pollFn: (targetId: string) => Promise<void>,
): UsePollingReturn {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollFnRef = useRef(pollFn);

  // Keep the callback ref fresh without restarting the interval
  useEffect(() => {
    pollFnRef.current = pollFn;
  }, [pollFn]);

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (targetId: string, intervalMs = 1000) => {
      stopPolling();
      intervalRef.current = setInterval(() => {
        pollFnRef.current(targetId);
      }, intervalMs);
    },
    [stopPolling],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, []);

  return { startPolling, stopPolling };
}
