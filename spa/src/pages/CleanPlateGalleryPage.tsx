/**
 * Clean-plate gallery: pick a capture run, browse a responsive grid of
 * lazy-loaded imgproxy thumbnails, and click a thumb to see the full presigned
 * image plus its metadata. Reads two backend endpoints:
 *
 *   GET /clean-plate/runs    — run picker
 *   GET /clean-plate/frames  — thumbnail grid (metadata + presigned/thumbnail URLs)
 */

import { useCallback, useEffect, useState } from 'react';
import { useApiFetch } from '../hooks/useAuthFetch';
import styles from './CleanPlateGalleryPage.module.css';

interface Run {
  run_id: string;
  frame_count: number;
  first_captured_at: string;
  last_captured_at: string;
}

interface Frame {
  id: string;
  run_id: string;
  camera_id: string;
  phase: string;
  pose_id: string;
  commanded_pan: number;
  commanded_tilt: number;
  commanded_zoom: number;
  burst_id: string | null;
  frame_index: number;
  captured_at: string;
  image_url: string;
  thumbnail_url: string;
  record: Record<string, unknown>;
}

interface FramesResponse {
  frames: Frame[];
  total: number;
  limit: number;
  offset: number;
}

interface RunsResponse {
  runs: Run[];
}

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export default function CleanPlateGalleryPage() {
  const apiFetch = useApiFetch();

  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>('');
  const [frames, setFrames] = useState<Frame[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingFrames, setLoadingFrames] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFrame, setSelectedFrame] = useState<Frame | null>(null);

  const fetchJson = useCallback(
    async <T,>(url: string, signal: AbortSignal): Promise<T> => {
      const res = await apiFetch(url, { signal });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      return (await res.json()) as T;
    },
    [apiFetch],
  );

  // Load runs once on mount; pre-select the most recent.
  useEffect(() => {
    const controller = new AbortController();

    void (async () => {
      setLoadingRuns(true);
      setError(null);
      try {
        const data = await fetchJson<RunsResponse>('/clean-plate/runs', controller.signal);
        if (controller.signal.aborted) return;
        setRuns(data.runs);
        if (data.runs.length > 0) setSelectedRunId(data.runs[0].run_id);
      } catch (err: unknown) {
        if (!controller.signal.aborted)
          setError(err instanceof Error ? err.message : 'Failed to load runs');
      } finally {
        if (!controller.signal.aborted) setLoadingRuns(false);
      }
    })();

    return () => controller.abort();
  }, [fetchJson]);

  // Load frames whenever the selected run changes.
  useEffect(() => {
    const controller = new AbortController();

    void (async () => {
      if (!selectedRunId) {
        setFrames([]);
        return;
      }
      setLoadingFrames(true);
      setError(null);
      try {
        const query = new URLSearchParams({ run_id: selectedRunId, limit: '200' });
        const data = await fetchJson<FramesResponse>(
          `/clean-plate/frames?${query.toString()}`,
          controller.signal,
        );
        if (!controller.signal.aborted) setFrames(data.frames);
      } catch (err: unknown) {
        if (!controller.signal.aborted)
          setError(err instanceof Error ? err.message : 'Failed to load frames');
      } finally {
        if (!controller.signal.aborted) setLoadingFrames(false);
      }
    })();

    return () => controller.abort();
  }, [selectedRunId, fetchJson]);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Clean-Plate Gallery</h1>
        <div className={styles.controls}>
          <label htmlFor="run-select" className={styles.label}>
            Run
          </label>
          <select
            id="run-select"
            className={styles.select}
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            disabled={loadingRuns || runs.length === 0}
          >
            {runs.length === 0 && <option value="">No runs</option>}
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id} ({run.frame_count} frames ·{' '}
                {formatTimestamp(run.last_captured_at)})
              </option>
            ))}
          </select>
        </div>
      </header>

      {error && <div className={styles.error}>{error}</div>}
      {loadingRuns && <p className={styles.status}>Loading runs…</p>}
      {!loadingRuns && loadingFrames && <p className={styles.status}>Loading frames…</p>}
      {!loadingFrames && selectedRunId && frames.length === 0 && !error && (
        <p className={styles.status}>No frames for this run.</p>
      )}

      <div className={styles.grid}>
        {frames.map((frame) => (
          <button
            type="button"
            key={frame.id}
            className={styles.thumb}
            onClick={() => setSelectedFrame(frame)}
            title={`${frame.pose_id} · ${formatTimestamp(frame.captured_at)}`}
          >
            <img
              className={styles.thumbImg}
              src={frame.thumbnail_url}
              alt={`Frame ${frame.id} (${frame.pose_id})`}
              loading="lazy"
            />
            <span className={styles.thumbCaption}>{frame.pose_id || frame.id}</span>
          </button>
        ))}
      </div>

      {selectedFrame && (
        <div
          className={styles.modalOverlay}
          role="presentation"
          onClick={() => setSelectedFrame(null)}
        >
          <div
            className={styles.modal}
            role="dialog"
            aria-modal="true"
            aria-label={`Frame ${selectedFrame.id}`}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              className={styles.closeBtn}
              onClick={() => setSelectedFrame(null)}
              aria-label="Close"
            >
              ×
            </button>
            <img
              className={styles.modalImg}
              src={selectedFrame.image_url}
              alt={`Frame ${selectedFrame.id} (full resolution)`}
            />
            <dl className={styles.meta}>
              <dt>Capture ID</dt>
              <dd>{selectedFrame.id}</dd>
              <dt>Pose</dt>
              <dd>{selectedFrame.pose_id || '—'}</dd>
              <dt>Camera</dt>
              <dd>{selectedFrame.camera_id || '—'}</dd>
              <dt>Phase</dt>
              <dd>{selectedFrame.phase || '—'}</dd>
              <dt>Pan / Tilt / Zoom</dt>
              <dd>
                {selectedFrame.commanded_pan} / {selectedFrame.commanded_tilt} /{' '}
                {selectedFrame.commanded_zoom}
              </dd>
              <dt>Captured at</dt>
              <dd>{formatTimestamp(selectedFrame.captured_at)}</dd>
              <dt>Frame index</dt>
              <dd>{selectedFrame.frame_index}</dd>
            </dl>
            <details className={styles.rawMeta}>
              <summary>Full metadata (optics, …)</summary>
              <pre className={styles.rawMetaPre}>
                {JSON.stringify(selectedFrame.record, null, 2)}
              </pre>
            </details>
          </div>
        </div>
      )}
    </div>
  );
}
