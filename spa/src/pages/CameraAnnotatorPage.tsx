import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react';
import client from '../api/client';
import { useAuth } from '../contexts/AuthContext';
import { useTenant } from '../contexts/TenantContext';
import styles from './CameraAnnotatorPage.module.css';

/* ------------------------------------------------------------------ */
/*  Types                                                             */
/* ------------------------------------------------------------------ */

interface Gcp {
  id: string;
  pixel_x: number;
  pixel_y: number;
}

interface Annotation {
  gcp_id: string;
  pixel_x: number;
  pixel_y: number;
}

interface PendingPoint {
  x: number;
  y: number;
}

interface DragState {
  index: number;
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export default function CameraAnnotatorPage() {
  const { credentials } = useAuth();
  const { selectedTenantId } = useTenant();

  /* ---- data state ---- */
  const [gcps, setGcps] = useState<Gcp[]>([]);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [images, setImages] = useState<string[]>([]);
  const [currentFilename, setCurrentFilename] = useState<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);

  /* ---- UI state ---- */
  const [pendingPoint, setPendingPoint] = useState<PendingPoint | null>(null);
  const [gcpSearchQuery, setGcpSearchQuery] = useState('');
  const [selectedGcpIndex, setSelectedGcpIndex] = useState(-1);
  const [coordsText, setCoordsText] = useState('Move mouse over image');
  const [saveLabel, setSaveLabel] = useState('Save to Repo');
  const [saving, setSaving] = useState(false);

  /* ---- drag state (ref to avoid re-renders during drag) ---- */
  const dragRef = useRef<DragState | null>(null);
  const annotationsRef = useRef(annotations);
  annotationsRef.current = annotations;

  /* ---- DOM refs ---- */
  const imgRef = useRef<HTMLImageElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const gcpSearchRef = useRef<HTMLInputElement>(null);

  /* ---- derived ---- */
  const usedGcpIds = new Set(annotations.map((a) => a.gcp_id));
  const filteredGcps = gcps
    .filter((g) => g.id.toLowerCase().includes(gcpSearchQuery.toLowerCase()))
    .filter((g) => !usedGcpIds.has(g.id));
  const safeSelectedIndex =
    selectedGcpIndex >= 0 && selectedGcpIndex < filteredGcps.length
      ? selectedGcpIndex
      : filteredGcps.length > 0
        ? 0
        : -1;

  /* ================================================================ */
  /*  Image loading via authenticated fetch                           */
  /* ================================================================ */

  const loadImage = useCallback(
    async (filename: string) => {
      if (!credentials || !selectedTenantId) return;
      const encoded = btoa(
        `${credentials.username}:${credentials.password}`,
      );
      const params = new URLSearchParams({
        tenant_id: selectedTenantId,
        image_filename: filename,
        t: String(Date.now()),
      });
      try {
        const resp = await fetch(
          `/camera-annotator/image/?${params.toString()}`,
          { headers: { Authorization: `Basic ${encoded}` } },
        );
        if (!resp.ok) return;
        const blob = await resp.blob();
        setImageUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return URL.createObjectURL(blob);
        });
      } catch {
        /* ignore network errors */
      }
    },
    [credentials, selectedTenantId],
  );

  /* ================================================================ */
  /*  Data loading                                                    */
  /* ================================================================ */

  /* Load GCPs */
  useEffect(() => {
    if (!selectedTenantId) return;
    let cancelled = false;

    (async () => {
      const { data } = await client.GET('/camera-annotator/api/gcps/', {
        params: { query: { tenant_id: selectedTenantId } },
      });
      if (!cancelled && data) setGcps(data);
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedTenantId]);

  /* Load available images */
  useEffect(() => {
    if (!selectedTenantId) return;
    let cancelled = false;

    (async () => {
      const { data } = await client.GET('/camera-annotator/api/images/', {
        params: { query: { tenant_id: selectedTenantId } },
      });
      if (!cancelled && data) {
        setImages(data);
        if (data.length > 0 && !currentFilename) {
          setCurrentFilename(data[0]);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // currentFilename intentionally omitted — we only auto-select on first load
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTenantId]);

  /* Load annotations + image when currentFilename changes */
  useEffect(() => {
    if (!selectedTenantId || !currentFilename) return;
    let cancelled = false;

    (async () => {
      const { data } = await client.GET(
        '/camera-annotator/api/annotations/',
        {
          params: {
            query: {
              tenant_id: selectedTenantId,
              image_filename: currentFilename,
            },
          },
        },
      );
      if (!cancelled && data) setAnnotations(data);
    })();

    loadImage(currentFilename);

    return () => {
      cancelled = true;
    };
  }, [selectedTenantId, currentFilename, loadImage]);

  /* Revoke object URL on unmount */
  useEffect(() => {
    return () => {
      setImageUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
    };
  }, []);

  /* ================================================================ */
  /*  Image click -> pending point                                    */
  /* ================================================================ */

  const handleImageClick = useCallback(
    (e: ReactMouseEvent<HTMLImageElement>) => {
      if (dragRef.current) return;
      const img = imgRef.current;
      if (!img) return;

      const rect = img.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      setPendingPoint({ x, y });
      setGcpSearchQuery('');
      setSelectedGcpIndex(0);

      // Focus the search input after React renders
      requestAnimationFrame(() => gcpSearchRef.current?.focus());
    },
    [],
  );

  /* ================================================================ */
  /*  GCP selection                                                   */
  /* ================================================================ */

  const selectGcp = useCallback(
    (gcp: Gcp) => {
      if (!pendingPoint) return;

      setAnnotations((prev) => [
        ...prev,
        { pixel_x: pendingPoint.x, pixel_y: pendingPoint.y, gcp_id: gcp.id },
      ]);
      setPendingPoint(null);
      setGcpSearchQuery('');
    },
    [pendingPoint],
  );

  const handleGcpKeyDown = useCallback(
    (e: ReactKeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedGcpIndex((i) =>
          i < filteredGcps.length - 1 ? i + 1 : i,
        );
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedGcpIndex((i) => (i > 0 ? i - 1 : i));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (safeSelectedIndex >= 0 && filteredGcps[safeSelectedIndex]) {
          selectGcp(filteredGcps[safeSelectedIndex]);
        }
      } else if (e.key === 'Escape') {
        setPendingPoint(null);
        setGcpSearchQuery('');
      }
    },
    [filteredGcps, safeSelectedIndex, selectGcp],
  );

  /* ================================================================ */
  /*  Marker dragging                                                 */
  /* ================================================================ */

  const handleMarkerMouseDown = useCallback(
    (index: number, e: ReactMouseEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      dragRef.current = { index };
      // Force a re-render to apply dragging class
      setAnnotations((a) => [...a]);
    },
    [],
  );

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const img = imgRef.current;
      if (!img) return;
      const rect = img.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (dragRef.current) {
        e.preventDefault();
        const idx = dragRef.current.index;
        const ann = annotationsRef.current[idx];
        if (!ann) return;

        const clampedX = Math.max(0, Math.min(img.naturalWidth, x));
        const clampedY = Math.max(0, Math.min(img.naturalHeight, y));

        // Update annotation in place and re-render
        setAnnotations((prev) =>
          prev.map((a, i) =>
            i === idx ? { ...a, pixel_x: clampedX, pixel_y: clampedY } : a,
          ),
        );
        setCoordsText(
          `Dragging ${ann.gcp_id}: (${clampedX.toFixed(1)}, ${clampedY.toFixed(1)})`,
        );
      } else {
        setCoordsText(`Pixel: (${x.toFixed(1)}, ${y.toFixed(1)})`);
      }
    };

    const handleMouseUp = () => {
      if (dragRef.current) {
        dragRef.current = null;
        // Force re-render to remove dragging class
        setAnnotations((a) => [...a]);
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  /* ================================================================ */
  /*  Delete annotation                                               */
  /* ================================================================ */

  const deleteAnnotation = useCallback((index: number) => {
    setAnnotations((prev) => prev.filter((_, i) => i !== index));
  }, []);

  /* ================================================================ */
  /*  Switch image                                                    */
  /* ================================================================ */

  const handleImageSwitch = useCallback(
    async (filename: string) => {
      if (!selectedTenantId) return;

      const { data } = await client.POST(
        '/camera-annotator/api/switch-image/',
        {
          params: { query: { tenant_id: selectedTenantId } },
          body: { filename },
        },
      );

      if (data?.success) {
        setPendingPoint(null);
        setGcpSearchQuery('');
        setAnnotations(data.annotations);
        setCurrentFilename(data.filename);
        loadImage(data.filename);
      }
    },
    [selectedTenantId, loadImage],
  );

  /* ================================================================ */
  /*  Save annotations                                                */
  /* ================================================================ */

  const handleSave = useCallback(async () => {
    if (
      annotations.length === 0 ||
      !selectedTenantId ||
      !currentFilename ||
      saving
    )
      return;

    setSaving(true);
    const cleanAnnotations = annotations.map((a) => ({
      gcp_id: a.gcp_id,
      pixel_x: parseFloat(a.pixel_x.toFixed(1)),
      pixel_y: parseFloat(a.pixel_y.toFixed(1)),
    }));

    try {
      const { data } = await client.POST(
        '/camera-annotator/api/save-annotations/',
        {
          params: { query: { tenant_id: selectedTenantId } },
          body: {
            image_filename: currentFilename,
            annotations: cleanAnnotations,
          },
        },
      );

      if (data?.success) {
        setSaveLabel(`Saved ${data.saved}!`);
        setTimeout(() => setSaveLabel('Save to Repo'), 2000);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      alert(`Save failed: ${message}`);
    } finally {
      setSaving(false);
    }
  }, [annotations, selectedTenantId, currentFilename, saving]);

  /* ================================================================ */
  /*  Render                                                          */
  /* ================================================================ */

  if (!selectedTenantId) {
    return <div className={styles.loading}>Select a tenant to begin.</div>;
  }

  return (
    <div className={styles.wrapper}>
      {/* ---- Left panel: Image ---- */}
      <div className={styles.imagePanel}>
        <div className={styles.imageContainer}>
          <div className={styles.imageWrapper} ref={wrapperRef}>
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
                {currentFilename ? 'Loading image...' : 'No image selected'}
              </div>
            )}

            {/* Annotation markers */}
            {annotations.map((ann, i) => (
              <div
                key={`${ann.gcp_id}-${i}`}
                className={`${styles.marker}${dragRef.current?.index === i ? ` ${styles.markerDragging}` : ''}`}
                style={{ left: ann.pixel_x, top: ann.pixel_y }}
                onMouseDown={(e) => handleMarkerMouseDown(i, e)}
              >
                <span className={styles.markerLabel}>{ann.gcp_id}</span>
              </div>
            ))}

            {/* Pending marker */}
            {pendingPoint && (
              <div
                className={`${styles.marker} ${styles.pending}`}
                style={{ left: pendingPoint.x, top: pendingPoint.y }}
              >
                <span className={styles.markerLabel}>?</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ---- Right panel: Controls ---- */}
      <div className={styles.controlPanel}>
        {/* Image selector */}
        <div className={styles.panelSection}>
          <h2>Select Image</h2>
          <select
            className={styles.imageSelector}
            value={currentFilename ?? ''}
            onChange={(e) => handleImageSwitch(e.target.value)}
          >
            {images.map((filename) => (
              <option key={filename} value={filename}>
                {filename}
              </option>
            ))}
          </select>
        </div>

        {/* Camera frame info */}
        <div className={styles.panelSection}>
          <h2>Camera Frame</h2>
          <div className={styles.filename}>
            {currentFilename ?? 'No image selected'}
          </div>
          <div className={styles.instructions}>
            Click on the image to add annotation points. Drag markers to
            adjust. Type to filter GCPs.
          </div>
        </div>

        {/* GCP input section (visible when pending point exists) */}
        <div
          className={
            pendingPoint
              ? styles.gcpInputSectionActive
              : styles.gcpInputSection
          }
        >
          <h2>Select GCP for Point</h2>
          <input
            ref={gcpSearchRef}
            type="text"
            className={styles.gcpSearch}
            placeholder="Type to filter GCPs..."
            autoComplete="off"
            value={gcpSearchQuery}
            onChange={(e) => {
              setGcpSearchQuery(e.target.value);
              setSelectedGcpIndex(0);
            }}
            onKeyDown={handleGcpKeyDown}
          />
          <div className={styles.gcpDropdown}>
            {filteredGcps.length === 0 ? (
              <div className={styles.emptyText}>No matching GCPs</div>
            ) : (
              filteredGcps.map((gcp, i) => (
                <div
                  key={gcp.id}
                  className={`${styles.gcpOption}${i === safeSelectedIndex ? ` ${styles.gcpOptionSelected}` : ''}`}
                  onClick={() => selectGcp(gcp)}
                >
                  <span className={styles.gcpOptionId}>{gcp.id}</span>{' '}
                  <span className={styles.gcpOptionCoords}>
                    ({gcp.pixel_x.toFixed(1)}, {gcp.pixel_y.toFixed(1)})
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Annotations header */}
        <div className={styles.panelSection}>
          <h2>Annotations ({annotations.length})</h2>
        </div>

        {/* Annotations list */}
        <div className={styles.annotationsList}>
          {annotations.map((ann, i) => (
            <div key={`${ann.gcp_id}-${i}`} className={styles.annotationItem}>
              <div className={styles.annotationInfo}>
                <span className={styles.annotationGcpId}>{ann.gcp_id}</span>{' '}
                <span className={styles.annotationPixel}>
                  ({ann.pixel_x.toFixed(1)}, {ann.pixel_y.toFixed(1)})
                </span>
              </div>
              <button
                type="button"
                className={styles.deleteBtn}
                onClick={() => deleteAnnotation(i)}
              >
                Delete
              </button>
            </div>
          ))}
        </div>

        {/* Save section */}
        <div className={styles.panelSection}>
          <div className={styles.saveActions}>
            <button
              type="button"
              className={styles.saveBtn}
              onClick={handleSave}
              disabled={annotations.length === 0 || saving}
            >
              {saveLabel}
            </button>
          </div>
        </div>
      </div>

      {/* Coordinates display */}
      <div className={styles.coordsDisplay}>{coordsText}</div>
    </div>
  );
}
