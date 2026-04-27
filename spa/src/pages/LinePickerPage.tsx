import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Link } from 'react-router-dom';
import OpenSeadragon from 'openseadragon';
import OpenSeadragonViewer, {
  type TileSourceConfig,
} from '../components/OpenSeadragonViewer';
import LineOverlay, { type LineData } from '../components/LineOverlay';
import { setupDraggableOverlay } from '../hooks/useOpenSeadragonDrag';
import { useTenant } from '../contexts/TenantContext';
import client from '../api/client';
import type { components } from '../api/schema';
import styles from './LinePickerPage.module.css';

// ------------------------------------------------------------------ types

type ImageInfo = components['schemas']['ImageInfoResponse'];
type LineOut = components['schemas']['LineOut'];

interface Point {
  x: number;
  y: number;
}

// --------------------------------------------------------- helper: debounce

function debounce<T extends (...args: never[]) => void>(fn: T, ms: number): T {
  let timer: ReturnType<typeof setTimeout>;
  return ((...args: Parameters<T>) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  }) as unknown as T;
}

// ============================================================== component

export default function LinePickerPage() {
  const { selectedTenantId } = useTenant();

  // ---- server data ----
  const [imageInfo, setImageInfo] = useState<ImageInfo | null>(null);
  const [lines, setLines] = useState<LineOut[]>([]);
  const [nextId, setNextId] = useState<string>('L1');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // ---- viewer ----
  const [viewer, setViewer] = useState<OpenSeadragon.Viewer | null>(null);

  // ---- UI state ----
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);
  const [startPoint, setStartPoint] = useState<Point | null>(null);
  const [coordsText, setCoordsText] = useState('Click on image...');

  // ---- overlay DOM refs (endpoint markers + labels) ----
  // Keyed by lineId => { start: HTMLElement, end: HTMLElement }
  const endpointRefs = useRef<
    Record<string, { start: HTMLElement; end: HTMLElement }>
  >({});
  const labelRefs = useRef<Record<string, HTMLElement>>({});
  const trackerRefs = useRef<
    Record<string, { start: OpenSeadragon.MouseTracker; end: OpenSeadragon.MouseTracker }>
  >({});
  // Track pending marker element
  const pendingMarkerRef = useRef<HTMLElement | null>(null);
  // Dragging state
  const draggingRef = useRef<{
    lineId: string;
    which: 'start' | 'end';
    marker: HTMLElement;
  } | null>(null);

  // ---- mutable ref for lines (for use inside event handlers) ----
  const linesRef = useRef(lines);
  useEffect(() => {
    linesRef.current = lines;
  }, [lines]);

  const selectedLineIdRef = useRef(selectedLineId);
  useEffect(() => {
    selectedLineIdRef.current = selectedLineId;
  }, [selectedLineId]);

  // ================================================================ API calls

  const fetchImageInfo = useCallback(async () => {
    const { data, error: err } = await client.GET(
      '/line-picker/api/image/info/',
      { params: { query: { tenant_id: '' } } },
    );
    if (err || !data) throw new Error('Failed to load image info');
    return data;
  }, []);

  const fetchLines = useCallback(async () => {
    const { data, error: err } = await client.GET(
      '/line-picker/api/lines/',
      { params: { query: { tenant_id: '' } } },
    );
    if (err || !data) throw new Error('Failed to load lines');
    return data.lines;
  }, []);

  const fetchNextId = useCallback(async () => {
    const { data, error: err } = await client.GET(
      '/line-picker/api/lines/next-id/',
      { params: { query: { tenant_id: '' } } },
    );
    if (err || !data) return 'L?';
    return data.next_id;
  }, []);

  const apiCreateLine = useCallback(
    async (start: Point, end: Point) => {
      const { data, error: err } = await client.POST(
        '/line-picker/api/lines/',
        {
          params: { query: { tenant_id: '' } },
          body: {
            start_x: start.x,
            start_y: start.y,
            end_x: end.x,
            end_y: end.y,
          },
        },
      );
      if (err || !data) throw new Error('Failed to create line');
      return data;
    },
    [],
  );

  const apiUpdateLine = useCallback(
    async (
      lineId: string,
      payload: { start_x?: number; start_y?: number; end_x?: number; end_y?: number },
    ) => {
      const { data, error: err } = await client.PUT(
        '/line-picker/api/lines/{line_id}/',
        {
          params: { query: { tenant_id: '' }, path: { line_id: lineId } },
          body: payload,
        },
      );
      if (err || !data) throw new Error('Failed to update line');
      return data;
    },
    [],
  );

  const apiDeleteLine = useCallback(async (lineId: string) => {
    const { error: err } = await client.DELETE(
      '/line-picker/api/lines/{line_id}/',
      { params: { query: { tenant_id: '' }, path: { line_id: lineId } } },
    );
    if (err) throw new Error('Failed to delete line');
  }, []);

  const fetchGeoCoords = useCallback(
    async (px: number, py: number) => {
      const { data } = await client.GET(
        '/line-picker/api/geo-coords/',
        { params: { query: { pixel_x: px, pixel_y: py, tenant_id: '' } } },
      );
      return data ?? null;
    },
    [],
  );

  // ========================================================= overlay helpers

  /** Remove all endpoint markers + labels from the viewer. */
  const clearOverlays = useCallback(() => {
    if (!viewer) return;
    // Destroy mouse trackers
    for (const t of Object.values(trackerRefs.current)) {
      t.start.destroy();
      t.end.destroy();
    }
    trackerRefs.current = {};
    // Remove endpoint overlays
    for (const ep of Object.values(endpointRefs.current)) {
      try { viewer.removeOverlay(ep.start); } catch { /* already removed */ }
      try { viewer.removeOverlay(ep.end); } catch { /* already removed */ }
    }
    endpointRefs.current = {};
    // Remove label overlays
    for (const el of Object.values(labelRefs.current)) {
      try { viewer.removeOverlay(el); } catch { /* already removed */ }
    }
    labelRefs.current = {};
  }, [viewer]);

  /** Add endpoint markers + draggable behavior for a single line. */
  const addLineOverlays = useCallback(
    (line: LineOut) => {
      if (!viewer || !imageInfo) return;

      // --- endpoint markers ---
      const startEl = document.createElement('div');
      startEl.className = 'endpoint-marker start';
      startEl.title = `${line.line_id} start`;

      const endEl = document.createElement('div');
      endEl.className = 'endpoint-marker end';
      endEl.title = `${line.line_id} end`;

      viewer.addOverlay({
        element: startEl,
        location: viewer.viewport.imageToViewportCoordinates(
          line.start_x,
          line.start_y,
        ),
        placement: OpenSeadragon.Placement.CENTER,
      });
      viewer.addOverlay({
        element: endEl,
        location: viewer.viewport.imageToViewportCoordinates(
          line.end_x,
          line.end_y,
        ),
        placement: OpenSeadragon.Placement.CENTER,
      });

      endpointRefs.current[line.line_id] = { start: startEl, end: endEl };

      // --- label at midpoint ---
      const labelEl = document.createElement('div');
      labelEl.className = 'line-label';
      labelEl.textContent = line.line_id;
      labelEl.title = `Click to select ${line.line_id}`;
      const midX = (line.start_x + line.end_x) / 2;
      const midY = (line.start_y + line.end_y) / 2;
      viewer.addOverlay({
        element: labelEl,
        location: viewer.viewport.imageToViewportCoordinates(midX, midY),
        placement: OpenSeadragon.Placement.BOTTOM,
      });
      labelEl.addEventListener('click', (e) => {
        e.stopPropagation();
        selectLine(line.line_id);
      });
      labelRefs.current[line.line_id] = labelEl;

      // --- draggable endpoints ---
      const makeDrag = (el: HTMLElement, which: 'start' | 'end') =>
        setupDraggableOverlay(viewer, el, imageInfo, {
          onDragStart: (elem) => {
            draggingRef.current = {
              lineId: line.line_id,
              which,
              marker: elem,
            };
            elem.classList.add('adjusting');
          },
          onDrag: (x, y, elem) => {
            viewer.updateOverlay(
              elem,
              viewer.viewport.imageToViewportCoordinates(x, y),
            );
          },
          onDragEnd: async (x, y, elem) => {
            elem.classList.remove('adjusting');
            draggingRef.current = null;
            const payload =
              which === 'start'
                ? { start_x: x, start_y: y }
                : { end_x: x, end_y: y };
            try {
              const updated = await apiUpdateLine(line.line_id, payload);
              setLines((prev) =>
                prev.map((l) =>
                  l.line_id === line.line_id ? updated : l,
                ),
              );
            } catch {
              // Reload on failure to reset positions
              const fresh = await fetchLines();
              setLines(fresh);
            }
          },
        });

      const startTracker = makeDrag(startEl, 'start');
      const endTracker = makeDrag(endEl, 'end');
      trackerRefs.current[line.line_id] = {
        start: startTracker,
        end: endTracker,
      };
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [viewer, imageInfo, apiUpdateLine, fetchLines],
  );

  /** Rebuild all overlays from the current lines array. */
  const syncOverlays = useCallback(
    (lineArr: LineOut[]) => {
      clearOverlays();
      lineArr.forEach(addLineOverlays);
    },
    [clearOverlays, addLineOverlays],
  );

  // ============================================================ initial load

  useEffect(() => {
    if (!selectedTenantId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const [info, lineList, nid] = await Promise.all([
          fetchImageInfo(),
          fetchLines(),
          fetchNextId(),
        ]);
        if (cancelled) return;
        setImageInfo(info);
        setLines(lineList);
        setNextId(nid);
      } catch (e) {
        if (!cancelled)
          setError(e instanceof Error ? e.message : 'Initialization failed');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [selectedTenantId, fetchImageInfo, fetchLines, fetchNextId]);

  // ---- rebuild overlays whenever lines change AND viewer is ready ----
  useEffect(() => {
    if (viewer && imageInfo) {
      syncOverlays(lines);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lines, viewer, imageInfo]);

  // ---- update label selection styling ----
  useEffect(() => {
    for (const [id, el] of Object.entries(labelRefs.current)) {
      el.classList.toggle('selected', id === selectedLineId);
    }
    for (const [id, ep] of Object.entries(endpointRefs.current)) {
      ep.start.classList.toggle('selected-line', id === selectedLineId);
      ep.end.classList.toggle('selected-line', id === selectedLineId);
    }
  }, [selectedLineId, lines]);

  // ================================================= tile source (memoised)

  const tileSource: TileSourceConfig | null = useMemo(() => {
    if (!imageInfo || !selectedTenantId) return null;
    return {
      width: imageInfo.width,
      height: imageInfo.height,
      tileSize: 256,
      getTileUrl: (level: number, x: number, y: number) =>
        `/line-picker/api/image/tile/?x=${x}&y=${y}&z=${level}&tenant_id=${encodeURIComponent(selectedTenantId)}`,
    };
  }, [imageInfo, selectedTenantId]);

  // =================================================== line overlay data

  const lineOverlayData: LineData[] = useMemo(
    () =>
      lines.map((l) => ({
        id: l.line_id,
        points: [
          { x: l.start_x, y: l.start_y },
          { x: l.end_x, y: l.end_y },
        ],
        color: '#2196f3',
        selected: l.line_id === selectedLineId,
      })),
    [lines, selectedLineId],
  );

  // ============================================================ interactions

  /** Select a line and pan to its center. */
  const selectLine = useCallback(
    (lineId: string) => {
      setSelectedLineId(lineId);
      if (!viewer) return;
      const line = linesRef.current.find((l) => l.line_id === lineId);
      if (line) {
        const cx = (line.start_x + line.end_x) / 2;
        const cy = (line.start_y + line.end_y) / 2;
        viewer.viewport.panTo(
          viewer.viewport.imageToViewportCoordinates(cx, cy),
          false,
        );
      }
    },
    [viewer],
  );

  /** Cancel line-creation mode, removing the pending marker. */
  const cancelCreation = useCallback(() => {
    if (pendingMarkerRef.current && viewer) {
      try {
        viewer.removeOverlay(pendingMarkerRef.current);
      } catch { /* already gone */ }
      pendingMarkerRef.current = null;
    }
    setStartPoint(null);
  }, [viewer]);

  /** Handle canvas click: place start or end point for line creation. */
  const handleCanvasClick = useCallback(
    async (imgX: number, imgY: number) => {
      if (draggingRef.current) return;
      if (!imageInfo) return;
      if (imgX < 0 || imgX >= imageInfo.width || imgY < 0 || imgY >= imageInfo.height) return;

      if (!startPoint) {
        // Place start point
        const pt: Point = { x: imgX, y: imgY };
        setStartPoint(pt);

        // Add pending marker
        if (viewer) {
          const el = document.createElement('div');
          el.className = 'endpoint-marker pending start';
          viewer.addOverlay({
            element: el,
            location: viewer.viewport.imageToViewportCoordinates(imgX, imgY),
            placement: OpenSeadragon.Placement.CENTER,
          });
          pendingMarkerRef.current = el;
        }
      } else {
        // Place end point and auto-create line
        const dist = Math.hypot(imgX - startPoint.x, imgY - startPoint.y);
        if (dist < 5) return; // too close

        try {
          await apiCreateLine(startPoint, { x: imgX, y: imgY });
          // Remove pending marker
          cancelCreation();
          // Reload
          const [freshLines, nid] = await Promise.all([
            fetchLines(),
            fetchNextId(),
          ]);
          setLines(freshLines);
          setNextId(nid);
        } catch (e) {
          console.error('Failed to create line:', e);
        }
      }
    },
    [startPoint, imageInfo, viewer, apiCreateLine, cancelCreation, fetchLines, fetchNextId],
  );

  /** Debounced coordinate display. */
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const handleMouseMove = useCallback(
    debounce(async (imgX: number, imgY: number) => {
      if (!imageInfo) return;
      if (
        imgX < 0 ||
        imgX >= imageInfo.width ||
        imgY < 0 ||
        imgY >= imageInfo.height
      ) {
        setCoordsText('Outside image');
        return;
      }

      let text = `Pixel: ${imgX.toFixed(1)}, ${imgY.toFixed(1)}`;

      if (imageInfo.geotransform) {
        try {
          const geo = await fetchGeoCoords(imgX, imgY);
          if (geo?.easting != null && geo.northing != null) {
            text += `\nGeo: ${geo.easting.toFixed(2)}, ${geo.northing.toFixed(2)}`;
            if (geo.crs) text += `\nCRS: ${geo.crs}`;
          }
        } catch {
          // silently ignore
        }
      }
      setCoordsText(text);
    }, 50),
    [imageInfo, fetchGeoCoords],
  );

  /** Delete a line. */
  const handleDeleteLine = useCallback(
    async (lineId: string) => {
      try {
        await apiDeleteLine(lineId);
        if (selectedLineIdRef.current === lineId) setSelectedLineId(null);
        const [freshLines, nid] = await Promise.all([
          fetchLines(),
          fetchNextId(),
        ]);
        setLines(freshLines);
        setNextId(nid);
      } catch (e) {
        console.error('Failed to delete line:', e);
      }
    },
    [apiDeleteLine, fetchLines, fetchNextId],
  );

  /** ESC key cancels creation. */
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && startPoint) cancelCreation();
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [startPoint, cancelCreation]);

  // ============================================================ render

  if (!selectedTenantId) {
    return <div className={styles.loading}>Select a tenant to continue.</div>;
  }
  if (loading) {
    return <div className={styles.loading}>Loading Line Picker...</div>;
  }
  if (error) {
    return <div className={styles.error}>Error: {error}</div>;
  }
  if (!imageInfo || !tileSource) {
    return <div className={styles.error}>No image data available.</div>;
  }

  const modeClass = startPoint ? styles.modeSelecting : styles.modeIdle;
  const modeText = startPoint
    ? 'Click to set end point (ESC to cancel)'
    : 'Click on map to set start point';

  return (
    <div className={styles.wrapper}>
      {/* ---------- SIDEBAR ---------- */}
      <div className={styles.sidebar}>
        <Link to="/" className={styles.backLink}>
          &larr; Back to Tools
        </Link>
        <h1 className={styles.title}>Line Picker</h1>
        <div className={styles.filename}>{imageInfo.filename}</div>

        {/* Line creation mode */}
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Line Creation Mode</div>
          <div className={`${styles.modeIndicator} ${modeClass}`}>
            {modeText}
          </div>
          {startPoint && (
            <div className={styles.selectedPoint}>
              <span className={styles.selectedPointLabel}>Start: </span>
              <span className={styles.selectedPointValue}>
                ({startPoint.x.toFixed(1)}, {startPoint.y.toFixed(1)})
              </span>
            </div>
          )}
          <div className={styles.nextId}>Next: {nextId}</div>
          <div className={styles.btnGroup}>
            <button
              type="button"
              className={`${styles.btn} ${styles.btnSecondary}`}
              disabled={!startPoint}
              onClick={cancelCreation}
            >
              Cancel
            </button>
          </div>
        </div>

        {/* Coordinates */}
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Coordinates</div>
          <div className={styles.coords}>{coordsText}</div>
        </div>

        {/* Lines list */}
        <div className={styles.section}>
          <div className={styles.sectionTitle}>
            Lines ({lines.length})
          </div>
          <div className={styles.lineList}>
            {lines.map((line) => {
              const isSelected = line.line_id === selectedLineId;
              const itemClass = [
                styles.lineItem,
                isSelected ? styles.lineItemSelected : '',
              ]
                .filter(Boolean)
                .join(' ');

              return (
                <div
                  key={line.line_id}
                  className={itemClass}
                  onClick={() => selectLine(line.line_id)}
                >
                  <div className={styles.lineInfo}>
                    <span className={styles.lineId}>{line.line_id}</span>
                    <span className={styles.lineCoords}>
                      ({line.start_x.toFixed(0)}, {line.start_y.toFixed(0)})
                      {' -> '}
                      ({line.end_x.toFixed(0)}, {line.end_y.toFixed(0)})
                    </span>
                  </div>
                  <button
                    type="button"
                    className={styles.deleteBtn}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteLine(line.line_id);
                    }}
                  >
                    X
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ---------- VIEWER ---------- */}
      <div className={styles.viewerArea}>
        <OpenSeadragonViewer
          tileSource={tileSource}
          onViewerReady={setViewer}
          onCanvasClick={handleCanvasClick}
          onMouseMove={handleMouseMove}
          className={styles.viewerArea}
        />
        <LineOverlay
          viewer={viewer}
          lines={lineOverlayData}
          onLineClick={selectLine}
        />
      </div>
    </div>
  );
}
