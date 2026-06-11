import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from 'react';
import client from '../api/client';
import { useTenant } from '../contexts/TenantContext';
import { useImageBlob } from '../hooks/useImageBlob';
import styles from './ClickToGpsPage.module.css';

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */

interface FrameSummary {
  name: string;
  image: string;
  annotation_count: number;
}

interface ClickedPoint {
  /** Pixel coordinates in the camera image (natural resolution). */
  pixelX: number;
  pixelY: number;
  latitude: number | null;
  longitude: number | null;
  confidence: number | null;
  onHorizon: boolean;
  error: string | null;
}

/* Confidence colour bands (DoD #33): green > 0.7, yellow 0.5–0.7, red < 0.5. */
function confidenceColor(point: ClickedPoint): string {
  if (point.error || point.confidence === null) return '#e53935'; // red
  if (point.confidence > 0.7) return '#43a047'; // green
  if (point.confidence >= 0.5) return '#fbc02d'; // yellow
  return '#e53935'; // red
}

function gpsLabel(point: ClickedPoint): string {
  if (point.latitude === null || point.longitude === null) return 'no fix';
  return `${point.latitude.toFixed(6)}, ${point.longitude.toFixed(6)}`;
}

const MAX_POINTS = 5;
const FLASH_MS = 3000;

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export default function ClickToGpsPage() {
  const { selectedTenantId } = useTenant();
  const { imageUrl, loadImage, clearImage } = useImageBlob();

  const [frames, setFrames] = useState<FrameSummary[]>([]);
  const [currentFrame, setCurrentFrame] = useState<FrameSummary | null>(null);
  const [clickedPoints, setClickedPoints] = useState<ClickedPoint[]>([]);
  const [flash, setFlash] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement>(null);
  const clickedRef = useRef<ClickedPoint[]>([]);
  clickedRef.current = clickedPoints;

  /* Select a frame and reset the per-frame marker state. */
  const selectFrame = useCallback((frame: FrameSummary | null) => {
    setCurrentFrame(frame);
    setClickedPoints([]);
    setFlash(null);
  }, []);

  /* ---- Load projectable frames on tenant change ---- */
  useEffect(() => {
    if (!selectedTenantId) return;
    const controller = new AbortController();

    (async () => {
      const { data } = await client.GET('/click-to-gps/api/frames/', {
        params: { query: { tenant_id: selectedTenantId } },
        signal: controller.signal,
      });
      if (controller.signal.aborted || !data) return;
      setFrames(data);
      selectFrame(data.length > 0 ? data[0] : null);
    })().catch((err: unknown) => {
      if (!controller.signal.aborted) console.error('Failed to load frames:', err);
    });

    return () => controller.abort();
  }, [selectedTenantId, selectFrame]);

  /* ---- Load the selected frame image ---- */
  useEffect(() => {
    if (!currentFrame) {
      clearImage();
      return;
    }
    loadImage('/click-to-gps/image/', currentFrame.image, true);
  }, [currentFrame, loadImage, clearImage]);

  /* ================================================================ */
  /*  Click → project                                                 */
  /* ================================================================ */

  const handleImageClick = useCallback(
    async (e: ReactMouseEvent<HTMLImageElement>) => {
      const img = imgRef.current;
      if (!img || !currentFrame || !selectedTenantId) return;

      // Map display coordinates to natural image pixels (scale-safe).
      const rect = img.getBoundingClientRect();
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      const pixelX = (e.clientX - rect.left) * scaleX;
      const pixelY = (e.clientY - rect.top) * scaleY;

      try {
        const { data, error } = await client.POST('/click-to-gps/api/project/', {
          params: { query: { tenant_id: selectedTenantId } },
          body: {
            test_case_name: currentFrame.name,
            pixel_x: pixelX,
            pixel_y: pixelY,
          },
        });

        if (error || !data) {
          console.error('Projection request failed:', error);
          return;
        }

        const point: ClickedPoint = {
          pixelX,
          pixelY,
          latitude: data.latitude ?? null,
          longitude: data.longitude ?? null,
          confidence: data.confidence ?? null,
          onHorizon: data.on_horizon,
          error: data.success ? null : (data.error ?? 'Projection failed'),
        };

        setClickedPoints((prev) => [...prev, point].slice(-MAX_POINTS));
        setFlash(point.error ? point.error : `GPS: ${gpsLabel(point)}`);
      } catch (err) {
        console.error('Projection request failed:', err);
      }
    },
    [currentFrame, selectedTenantId],
  );

  /* ---- Auto-clear the transient flash banner after 3 s ---- */
  useEffect(() => {
    if (flash === null) return;
    const id = window.setTimeout(() => setFlash(null), FLASH_MS);
    return () => window.clearTimeout(id);
  }, [flash]);

  useEffect(() => {
    if (copyStatus === null) return;
    const id = window.setTimeout(() => setCopyStatus(null), FLASH_MS);
    return () => window.clearTimeout(id);
  }, [copyStatus]);

  /* ================================================================ */
  /*  Keyboard shortcuts: 'c' copy last GPS, 'x' clear markers        */
  /* ================================================================ */

  const copyLastGps = useCallback(async () => {
    const points = clickedRef.current;
    const last = [...points].reverse().find((p) => p.latitude !== null);
    if (!last || last.latitude === null || last.longitude === null) {
      setCopyStatus('No GPS coordinate to copy');
      return;
    }
    const text = `${last.latitude.toFixed(6)}, ${last.longitude.toFixed(6)}`;
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus(`Copied: ${text}`);
    } catch {
      setCopyStatus('Clipboard unavailable');
    }
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'SELECT')) {
        return;
      }
      if (e.key === 'c' || e.key === 'C') {
        void copyLastGps();
      } else if (e.key === 'x' || e.key === 'X') {
        setClickedPoints([]);
        setFlash(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [copyLastGps]);

  /* ================================================================ */
  /*  Render                                                          */
  /* ================================================================ */

  if (!selectedTenantId) {
    return <div className={styles.loading}>Select a tenant to begin.</div>;
  }

  return (
    <div className={styles.wrapper}>
      {/* ---- Left: camera frame ---- */}
      <div className={styles.imagePanel}>
        <div className={styles.imageContainer}>
          <div className={styles.imageWrapper}>
            {imageUrl ? (
              <img
                ref={imgRef}
                className={styles.frameImage}
                src={imageUrl}
                alt="Camera Frame"
                onClick={handleImageClick}
                draggable={false}
              />
            ) : (
              <div className={styles.loading}>
                {currentFrame ? 'Loading image…' : 'No frame available'}
              </div>
            )}

            {/* Clicked-point crosshairs + GPS labels */}
            {clickedPoints.map((p, i) => {
              const color = confidenceColor(p);
              const newest = i === clickedPoints.length - 1;
              return (
                <div
                  key={`${p.pixelX}-${p.pixelY}-${i}`}
                  className={`${styles.marker}${newest ? ` ${styles.markerNewest}` : ''}`}
                  style={{ left: p.pixelX, top: p.pixelY, color }}
                >
                  <span className={styles.crosshair} style={{ borderColor: color }} />
                  <span className={styles.markerLabel} style={{ color }}>
                    {p.error ? p.error : gpsLabel(p)}
                    {p.confidence !== null && !p.error
                      ? ` (${(p.confidence * 100).toFixed(0)}%)`
                      : ''}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {flash && <div className={styles.flash}>{flash}</div>}
      </div>

      {/* ---- Right: controls ---- */}
      <div className={styles.controlPanel}>
        <div className={styles.panelSection}>
          <h2>Click-to-GPS</h2>
          <p className={styles.instructions}>
            Click anywhere on the camera frame to estimate its GPS coordinate.
            Markers are colour-coded by confidence (green &gt; 0.7, yellow
            0.5–0.7, red &lt; 0.5). Press <kbd>c</kbd> to copy the last GPS,{' '}
            <kbd>x</kbd> to clear markers.
          </p>
        </div>

        <div className={styles.panelSection}>
          <h2>Frame</h2>
          {frames.length === 0 ? (
            <div className={styles.emptyText}>
              No frames with ≥4 GCP annotations for this tenant.
            </div>
          ) : (
            <select
              className={styles.frameSelector}
              value={currentFrame?.name ?? ''}
              onChange={(e) => {
                const next = frames.find((f) => f.name === e.target.value);
                if (next) selectFrame(next);
              }}
            >
              {frames.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name} ({f.annotation_count} GCPs)
                </option>
              ))}
            </select>
          )}
        </div>

        <div className={styles.panelSection}>
          <h2>Clicked Points ({clickedPoints.length})</h2>
          <div className={styles.pointList}>
            {clickedPoints.length === 0 ? (
              <div className={styles.emptyText}>Click the frame to start.</div>
            ) : (
              [...clickedPoints].reverse().map((p, i) => (
                <div key={`row-${i}`} className={styles.pointRow}>
                  <span
                    className={styles.swatch}
                    style={{ background: confidenceColor(p) }}
                  />
                  <span className={styles.pointGps}>{gpsLabel(p)}</span>
                  <span className={styles.pointConf}>
                    {p.error
                      ? p.onHorizon
                        ? 'horizon'
                        : 'error'
                      : p.confidence !== null
                        ? `${(p.confidence * 100).toFixed(0)}%`
                        : ''}
                  </span>
                </div>
              ))
            )}
          </div>
          <button
            type="button"
            className={styles.copyButton}
            onClick={() => void copyLastGps()}
          >
            Copy last GPS (c)
          </button>
          <button
            type="button"
            className={styles.clearButton}
            onClick={() => {
              setClickedPoints([]);
              setFlash(null);
            }}
          >
            Clear markers (x)
          </button>
          {copyStatus && <div className={styles.copyStatus}>{copyStatus}</div>}
        </div>
      </div>
    </div>
  );
}
