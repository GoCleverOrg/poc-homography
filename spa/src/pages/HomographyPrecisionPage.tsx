import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import OpenSeadragon from 'openseadragon';
import { Link } from 'react-router-dom';
import OpenSeadragonViewer, {
  type TileSourceConfig,
} from '../components/OpenSeadragonViewer';
import LineOverlay, { type LineData } from '../components/LineOverlay';
import { useAuth } from '../contexts/AuthContext';
import { useTenant } from '../contexts/TenantContext';
import client from '../api/client';
import styles from './HomographyPrecisionPage.module.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ImageInfo {
  width: number;
  height: number;
  filename: string;
  geotransform?: number[] | null;
  crs?: string | null;
}

interface TestCaseSummary {
  name: string;
  image: string;
  annotation_count: number;
}

interface LineTestCaseSummary {
  name: string;
  image: string;
  line_annotation_count: number;
}

interface HomographyMetrics {
  num_gcps: number;
  num_inliers: number;
  inlier_ratio: number;
  mean_reproj_error: number;
  max_reproj_error: number;
  rmse: number;
}

interface PerPointError {
  gcp_id: string;
  error_px: number;
  camera_dx: number;
  camera_dy: number;
  map_dx: number;
  map_dy: number;
  camera_original: number[];
  camera_reprojected: number[];
  map_original: number[];
  map_projected: number[];
}

interface OverlayPoint {
  gcp_id: string;
  x: number;
  y: number;
}

interface PointOverlays {
  camera: {
    annotations: OverlayPoint[];
    reprojected_gcps: OverlayPoint[];
  };
  map: {
    gcps: OverlayPoint[];
    projected_annotations: OverlayPoint[];
  };
}

interface LineErrorMetrics {
  num_lines: number;
  mean_line_error: number;
  max_line_error: number;
}

interface PerLineError {
  line_id: string;
  error_px: number;
  start_error: number;
  end_error: number;
  map_start_error: number;
  map_end_error: number;
}

interface OverlayLine {
  line_id: string;
  start: number[];
  end: number[];
}

interface LineOverlays {
  camera: {
    annotations: OverlayLine[];
    reprojected_lines: OverlayLine[];
  };
  map: {
    gcp_lines: OverlayLine[];
    projected_lines: OverlayLine[];
  };
}

// Status message shape
type StatusType = 'info' | 'error' | 'success' | null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const API = '/homography-precision';

function getErrorClass(error: number): string {
  if (error < 5) return styles.errorLow;
  if (error < 10) return styles.errorMedium;
  return styles.errorHigh;
}

function getMetricClass(
  value: number,
  thresholds: { good: number; medium: number },
): string {
  if (value < thresholds.good) return styles.good;
  if (value < thresholds.medium) return styles.medium;
  return styles.bad;
}

/**
 * Build a tile URL that includes tenant_id and HTTP-Basic credentials so that
 * OpenSeadragon (which loads tiles as bare `<img>` elements) passes auth.
 */
function buildTileUrl(
  base: string,
  params: Record<string, string | number>,
  tenantId: string | null,
  credentials: { username: string; password: string } | null,
): string {
  const qs = Object.entries(params)
    .map(([k, v]) => `${k}=${encodeURIComponent(v)}`)
    .join('&');
  let url = `${base}?${qs}`;
  if (tenantId) url += `&tenant_id=${encodeURIComponent(tenantId)}`;

  // Embed credentials into the URL for OSD image requests.
  if (credentials) {
    const origin = window.location.origin;
    const full = new URL(url, origin);
    full.username = credentials.username;
    full.password = credentials.password;
    return full.toString();
  }
  return url;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Sortable per-point error table. */
function PointErrorTable({
  errors,
  unit,
  viewerType,
  selectedId,
  onRowClick,
}: {
  errors: { id: string; error: number; dx: number; dy: number }[];
  unit: string;
  viewerType: 'camera' | 'map';
  selectedId: string | null;
  onRowClick: (id: string, viewer: 'camera' | 'map') => void;
}) {
  const [sortKey, setSortKey] = useState<'id' | 'error' | 'dx' | 'dy'>(
    'error',
  );
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const toggleSort = (key: typeof sortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const sorted = useMemo(() => {
    const copy = [...errors];
    copy.sort((a, b) => {
      let cmp: number;
      if (sortKey === 'id') {
        cmp = a.id.localeCompare(b.id);
      } else {
        cmp = (a[sortKey] ?? 0) - (b[sortKey] ?? 0);
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return copy;
  }, [errors, sortKey, sortDir]);

  const decimals = unit === 'm' ? 3 : 1;

  // Axis bias summary
  let axisSummary = '';
  let axisClass = styles.axisSummary;
  if (errors.length > 0) {
    let totalAbsDx = 0;
    let totalAbsDy = 0;
    for (const e of errors) {
      totalAbsDx += Math.abs(e.dx);
      totalAbsDy += Math.abs(e.dy);
    }
    const total = totalAbsDx + totalAbsDy;
    const xPct = total > 0 ? (totalAbsDx / total) * 100 : 50;
    axisSummary = `(X: ${xPct.toFixed(0)}%, Y: ${(100 - xPct).toFixed(0)}%)`;
    if (xPct > 60) axisClass = styles.axisSummaryXBiased;
    else if (xPct < 40) axisClass = styles.axisSummaryYBiased;
  }

  if (errors.length === 0) {
    return (
      <table className={styles.errorTable}>
        <thead>
          <tr>
            <th>GCP</th>
            <th>Total</th>
            <th>dX</th>
            <th>dY</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td colSpan={4} className={styles.noData}>
              No data
            </td>
          </tr>
        </tbody>
      </table>
    );
  }

  return (
    <>
      <h2 className={styles.tableSectionTitle}>
        Per-Point Errors ({unit === 'm' ? 'meters' : 'pixels'}){' '}
        <span className={axisClass}>{axisSummary}</span>
      </h2>
      <div className={styles.errorTableContainer}>
        <table className={styles.errorTable}>
          <thead>
            <tr>
              <th onClick={() => toggleSort('id')}>GCP</th>
              <th onClick={() => toggleSort('error')}>Total</th>
              <th onClick={() => toggleSort('dx')}>dX</th>
              <th onClick={() => toggleSort('dy')}>dY</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((item) => {
              const errorClass =
                unit === 'm'
                  ? getMetricClass(item.error, { good: 1, medium: 2 })
                  : getErrorClass(item.error);
              return (
                <tr
                  key={item.id}
                  className={selectedId === item.id ? styles.selected : ''}
                  onClick={() => onRowClick(item.id, viewerType)}
                >
                  <td>{item.id}</td>
                  <td className={errorClass}>{item.error.toFixed(decimals)}</td>
                  <td className={item.dx >= 0 ? styles.positive : styles.negative}>
                    {(item.dx >= 0 ? '+' : '') + item.dx.toFixed(decimals)}
                  </td>
                  <td className={item.dy >= 0 ? styles.positive : styles.negative}>
                    {(item.dy >= 0 ? '+' : '') + item.dy.toFixed(decimals)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** Sortable per-line error table. */
function LineErrorTable({
  errors,
  unit,
  viewerType,
  selectedId,
  onRowClick,
}: {
  errors: { id: string; error: number; start: number; end: number }[];
  unit: string;
  viewerType: 'camera' | 'map';
  selectedId: string | null;
  onRowClick: (id: string, viewer: 'camera' | 'map') => void;
}) {
  const [sortKey, setSortKey] = useState<'id' | 'error' | 'start' | 'end'>(
    'error',
  );
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const toggleSort = (key: typeof sortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const sorted = useMemo(() => {
    const copy = [...errors];
    copy.sort((a, b) => {
      let cmp: number;
      if (sortKey === 'id') {
        cmp = a.id.localeCompare(b.id);
      } else {
        cmp = (a[sortKey] ?? 0) - (b[sortKey] ?? 0);
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return copy;
  }, [errors, sortKey, sortDir]);

  const decimals = unit === 'm' ? 3 : 1;

  if (errors.length === 0) {
    return (
      <table className={styles.errorTable}>
        <thead>
          <tr>
            <th>Line ID</th>
            <th>Total</th>
            <th>Start</th>
            <th>End</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td colSpan={4} className={styles.noData}>
              No data
            </td>
          </tr>
        </tbody>
      </table>
    );
  }

  return (
    <>
      <h2 className={styles.tableSectionTitle}>
        Per-Line Errors ({unit === 'm' ? 'meters' : 'pixels'})
      </h2>
      <div className={styles.errorTableContainer}>
        <table className={styles.errorTable}>
          <thead>
            <tr>
              <th onClick={() => toggleSort('id')}>Line ID</th>
              <th onClick={() => toggleSort('error')}>Total</th>
              <th onClick={() => toggleSort('start')}>Start</th>
              <th onClick={() => toggleSort('end')}>End</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((item) => {
              const errorClass =
                unit === 'm'
                  ? getMetricClass(item.error, { good: 1, medium: 2 })
                  : getErrorClass(item.error);
              return (
                <tr
                  key={item.id}
                  className={selectedId === item.id ? styles.selected : ''}
                  onClick={() => onRowClick(item.id, viewerType)}
                >
                  <td>{item.id}</td>
                  <td className={errorClass}>{item.error.toFixed(decimals)}</td>
                  <td className={getErrorClass(item.start)}>
                    {item.start.toFixed(decimals)}
                  </td>
                  <td className={getErrorClass(item.end)}>
                    {item.end.toFixed(decimals)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// Main page component
// ---------------------------------------------------------------------------

export default function HomographyPrecisionPage() {
  const { credentials } = useAuth();
  const { selectedTenantId } = useTenant();

  // --- test case lists ---
  const [testCases, setTestCases] = useState<TestCaseSummary[]>([]);
  const [lineTestCases, setLineTestCases] = useState<LineTestCaseSummary[]>([]);

  // --- selected test case ---
  const [selectedCase, setSelectedCase] = useState('');
  const [selectedLineCase, setSelectedLineCase] = useState('');
  const [useLineHomography, setUseLineHomography] = useState(true);

  // --- image info ---
  const [cameraInfo, setCameraInfo] = useState<ImageInfo | null>(null);
  const [mapInfo, setMapInfo] = useState<ImageInfo | null>(null);

  // --- the currently loaded test case name (for camera tile URLs) ---
  const currentCaseRef = useRef<string>('');

  // --- OSD viewer refs ---
  const cameraViewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const mapViewerRef = useRef<OpenSeadragon.Viewer | null>(null);

  // --- overlay data ---
  const [pointOverlays, setPointOverlays] = useState<PointOverlays | null>(
    null,
  );
  const [metrics, setMetrics] = useState<HomographyMetrics | null>(null);
  const [perPointErrors, setPerPointErrors] = useState<PerPointError[]>([]);

  const [lineOverlaysData, setLineOverlaysData] =
    useState<LineOverlays | null>(null);
  const [lineMetrics, setLineMetrics] = useState<LineErrorMetrics | null>(null);
  const [perLineErrors, setPerLineErrors] = useState<PerLineError[]>([]);

  // --- active mode: 'points' | 'lines' ---
  const [activeMode, setActiveMode] = useState<'points' | 'lines'>('points');

  // --- status ---
  const [statusMsg, setStatusMsg] = useState('');
  const [statusType, setStatusType] = useState<StatusType>(null);

  // --- selection ---
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
  const [selectedLineId, setSelectedLineId] = useState<string | null>(null);

  // Helpers to show/hide status
  const showStatus = useCallback((msg: string, type: StatusType) => {
    setStatusMsg(msg);
    setStatusType(type);
  }, []);

  // -----------------------------------------------------------------------
  // Fetch test case lists on mount / tenant change
  // -----------------------------------------------------------------------

  useEffect(() => {
    if (!selectedTenantId) return;

    // Point test cases
    client
      .GET('/homography-precision/api/test-cases/', {
        params: { query: { tenant_id: selectedTenantId } },
      })
      .then(({ data, error }) => {
        if (error) {
          console.error('Failed to load test cases', error);
          return;
        }
        if (data) setTestCases(data.test_cases as TestCaseSummary[]);
      });

    // Line test cases (may not exist)
    client
      .GET('/homography-precision/api/line-test-cases/', {
        params: { query: { tenant_id: selectedTenantId } },
      })
      .then(({ data }) => {
        if (data)
          setLineTestCases(data.test_cases as LineTestCaseSummary[]);
      })
      .catch(() => {
        /* silently ignore -- line test cases may not exist */
      });
  }, [selectedTenantId]);

  // -----------------------------------------------------------------------
  // Tile source configs (memoized so OSD only swaps when info changes)
  // -----------------------------------------------------------------------

  const cameraTileSource: TileSourceConfig | null = useMemo(() => {
    if (!cameraInfo) return null;
    return {
      width: cameraInfo.width,
      height: cameraInfo.height,
      tileSize: 256,
      getTileUrl: (level: number, x: number, y: number) =>
        buildTileUrl(
          `${API}/api/camera-tile/`,
          { case: currentCaseRef.current, x, y, z: level },
          selectedTenantId,
          credentials,
        ),
    };
  }, [cameraInfo, selectedTenantId, credentials]);

  const mapTileSource: TileSourceConfig | null = useMemo(() => {
    if (!mapInfo) return null;
    return {
      width: mapInfo.width,
      height: mapInfo.height,
      tileSize: 256,
      getTileUrl: (level: number, x: number, y: number) =>
        buildTileUrl(
          `${API}/api/map-tile/`,
          { x, y, z: level },
          selectedTenantId,
          credentials,
        ),
    };
  }, [mapInfo, selectedTenantId, credentials]);

  // -----------------------------------------------------------------------
  // OSD overlay management -- add pin markers whenever viewer + data change
  // -----------------------------------------------------------------------

  const cameraOverlayEls = useRef<HTMLDivElement[]>([]);
  const mapOverlayEls = useRef<HTMLDivElement[]>([]);

  /** Create a pin marker DOM element for OSD overlays. */
  const createPinEl = useCallback(
    (color: 'Blue' | 'Green' | 'Red', label: string, gcpId: string) => {
      const el = document.createElement('div');
      const colorClass =
        color === 'Blue'
          ? styles.markerBlue
          : color === 'Green'
            ? styles.markerGreen
            : styles.markerRed;
      el.className = `${styles.marker} ${colorClass}`;
      el.dataset.gcpId = gcpId;
      const span = document.createElement('span');
      span.textContent = label;
      el.appendChild(span);
      return el;
    },
    [],
  );

  const clearOsdOverlays = useCallback(
    (viewer: OpenSeadragon.Viewer | null, elList: HTMLDivElement[]) => {
      if (!viewer) return;
      for (const el of elList) {
        try {
          viewer.removeOverlay(el);
        } catch {
          /* ignore */
        }
      }
      elList.length = 0;
    },
    [],
  );

  // Render camera point overlays
  useEffect(() => {
    const viewer = cameraViewerRef.current;
    if (!viewer || !pointOverlays || activeMode !== 'points') return;

    // Wait for viewer to be open
    const doRender = () => {
      clearOsdOverlays(viewer, cameraOverlayEls.current);

      const cam = pointOverlays.camera;
      cam.annotations.forEach((pt, i) => {
        const reproj = cam.reprojected_gcps[i];
        // Blue pin = annotation
        const blueEl = createPinEl('Blue', pt.gcp_id, pt.gcp_id);
        const blueLoc = viewer.viewport.imageToViewportCoordinates(pt.x, pt.y);
        viewer.addOverlay({
          element: blueEl,
          location: blueLoc,
          placement: OpenSeadragon.Placement.BOTTOM,
        });
        cameraOverlayEls.current.push(blueEl);

        // Red pin = reprojected
        const redEl = createPinEl('Red', pt.gcp_id, pt.gcp_id);
        const redLoc = viewer.viewport.imageToViewportCoordinates(
          reproj.x,
          reproj.y,
        );
        viewer.addOverlay({
          element: redEl,
          location: redLoc,
          placement: OpenSeadragon.Placement.BOTTOM,
        });
        cameraOverlayEls.current.push(redEl);
      });
    };

    if (viewer.world.getItemCount() > 0) {
      doRender();
    } else {
      viewer.addOnceHandler('open', doRender);
    }
  }, [pointOverlays, activeMode, clearOsdOverlays, createPinEl]);

  // Render map point overlays
  useEffect(() => {
    const viewer = mapViewerRef.current;
    if (!viewer || !pointOverlays || activeMode !== 'points') return;

    const doRender = () => {
      clearOsdOverlays(viewer, mapOverlayEls.current);

      const m = pointOverlays.map;
      m.gcps.forEach((pt, i) => {
        const proj = m.projected_annotations[i];
        // Green pin = GCP
        const greenEl = createPinEl('Green', pt.gcp_id, pt.gcp_id);
        const greenLoc = viewer.viewport.imageToViewportCoordinates(
          pt.x,
          pt.y,
        );
        viewer.addOverlay({
          element: greenEl,
          location: greenLoc,
          placement: OpenSeadragon.Placement.BOTTOM,
        });
        mapOverlayEls.current.push(greenEl);

        // Red pin = projected
        const redEl = createPinEl('Red', pt.gcp_id, pt.gcp_id);
        const redLoc = viewer.viewport.imageToViewportCoordinates(
          proj.x,
          proj.y,
        );
        viewer.addOverlay({
          element: redEl,
          location: redLoc,
          placement: OpenSeadragon.Placement.BOTTOM,
        });
        mapOverlayEls.current.push(redEl);
      });
    };

    if (viewer.world.getItemCount() > 0) {
      doRender();
    } else {
      viewer.addOnceHandler('open', doRender);
    }
  }, [pointOverlays, activeMode, clearOsdOverlays, createPinEl]);

  // Highlight selected point marker
  useEffect(() => {
    const allMarkers = [
      ...cameraOverlayEls.current,
      ...mapOverlayEls.current,
    ];
    for (const el of allMarkers) {
      if (el.dataset.gcpId === selectedPointId) {
        el.classList.add(styles.markerHighlighted);
      } else {
        el.classList.remove(styles.markerHighlighted);
      }
    }
  }, [selectedPointId]);

  // -----------------------------------------------------------------------
  // Line overlay data for <LineOverlay>
  // -----------------------------------------------------------------------

  const cameraLines: LineData[] = useMemo(() => {
    if (!lineOverlaysData || activeMode !== 'lines') return [];
    const cam = lineOverlaysData.camera;
    const result: LineData[] = [];

    // Annotation lines (blue solid)
    for (const line of cam.annotations) {
      result.push({
        id: `ann-${line.line_id}`,
        points: [
          { x: line.start[0], y: line.start[1] },
          { x: line.end[0], y: line.end[1] },
        ],
        color: '#2196f3',
        selected: selectedLineId === line.line_id,
      });
    }
    // Reprojected lines (red dashed -- rendered solid since LineOverlay
    // does not support dashes, but distinct from blue)
    for (const line of cam.reprojected_lines) {
      result.push({
        id: `reproj-${line.line_id}`,
        points: [
          { x: line.start[0], y: line.start[1] },
          { x: line.end[0], y: line.end[1] },
        ],
        color: '#f44336',
        selected: selectedLineId === line.line_id,
      });
    }
    return result;
  }, [lineOverlaysData, activeMode, selectedLineId]);

  const mapLines: LineData[] = useMemo(() => {
    if (!lineOverlaysData || activeMode !== 'lines') return [];
    const m = lineOverlaysData.map;
    const result: LineData[] = [];

    // GCP lines (green solid)
    for (const line of m.gcp_lines) {
      result.push({
        id: `gcp-${line.line_id}`,
        points: [
          { x: line.start[0], y: line.start[1] },
          { x: line.end[0], y: line.end[1] },
        ],
        color: '#4caf50',
        selected: selectedLineId === line.line_id,
      });
    }
    // Projected lines (red dashed)
    for (const line of m.projected_lines) {
      result.push({
        id: `proj-${line.line_id}`,
        points: [
          { x: line.start[0], y: line.start[1] },
          { x: line.end[0], y: line.end[1] },
        ],
        color: '#f44336',
        selected: selectedLineId === line.line_id,
      });
    }
    return result;
  }, [lineOverlaysData, activeMode, selectedLineId]);

  // Error lines connecting annotation to reprojected (yellow) for point mode
  const cameraErrorLines: LineData[] = useMemo(() => {
    if (!pointOverlays || activeMode !== 'points') return [];
    const cam = pointOverlays.camera;
    return cam.annotations.map((ann, i) => {
      const reproj = cam.reprojected_gcps[i];
      return {
        id: `err-cam-${ann.gcp_id}`,
        points: [
          { x: ann.x, y: ann.y },
          { x: reproj.x, y: reproj.y },
        ],
        color: '#ffeb3b',
      };
    });
  }, [pointOverlays, activeMode]);

  const mapErrorLines: LineData[] = useMemo(() => {
    if (!pointOverlays || activeMode !== 'points') return [];
    const m = pointOverlays.map;
    return m.gcps.map((gcp, i) => {
      const proj = m.projected_annotations[i];
      return {
        id: `err-map-${gcp.gcp_id}`,
        points: [
          { x: gcp.x, y: gcp.y },
          { x: proj.x, y: proj.y },
        ],
        color: '#ffeb3b',
      };
    });
  }, [pointOverlays, activeMode]);

  // -----------------------------------------------------------------------
  // Derived table data
  // -----------------------------------------------------------------------

  const GSD = useMemo(() => {
    if (mapInfo?.geotransform) return Math.abs(mapInfo.geotransform[1]);
    return 0.15;
  }, [mapInfo]);

  const cameraPointTableData = useMemo(
    () =>
      perPointErrors.map((e) => ({
        id: e.gcp_id,
        error: e.error_px,
        dx: e.camera_dx,
        dy: e.camera_dy,
      })),
    [perPointErrors],
  );

  const mapPointTableData = useMemo(
    () =>
      perPointErrors.map((e) => ({
        id: e.gcp_id,
        error: Math.sqrt(e.map_dx ** 2 + e.map_dy ** 2) * GSD,
        dx: e.map_dx * GSD,
        dy: e.map_dy * GSD,
      })),
    [perPointErrors, GSD],
  );

  const cameraLineTableData = useMemo(
    () =>
      perLineErrors.map((e) => ({
        id: e.line_id,
        error: e.error_px,
        start: e.start_error,
        end: e.end_error,
      })),
    [perLineErrors],
  );

  const mapLineTableData = useMemo(
    () =>
      perLineErrors.map((e) => ({
        id: e.line_id,
        error: ((e.map_start_error + e.map_end_error) / 2) * GSD,
        start: e.map_start_error * GSD,
        end: e.map_end_error * GSD,
      })),
    [perLineErrors, GSD],
  );

  // -----------------------------------------------------------------------
  // Zoom-to-point / zoom-to-line helpers
  // -----------------------------------------------------------------------

  const zoomToPoint = useCallback(
    (viewer: OpenSeadragon.Viewer | null, x: number, y: number) => {
      if (!viewer) return;
      const zoomLevel = 5;
      const vpPt = viewer.viewport.imageToViewportCoordinates(x, y);
      const item = viewer.world.getItemAt(0);
      if (!item) return;
      const imageWidth = item.getContentSize().x;
      const viewWidth = imageWidth / zoomLevel;
      const halfWidth = viewWidth / 2 / imageWidth;
      const bounds = new OpenSeadragon.Rect(
        vpPt.x - halfWidth,
        vpPt.y - halfWidth,
        halfWidth * 2,
        halfWidth * 2,
      );
      viewer.viewport.fitBounds(bounds, false);
    },
    [],
  );

  const handlePointClick = useCallback(
    (gcpId: string, viewerType: 'camera' | 'map') => {
      setSelectedPointId(gcpId);
      setSelectedLineId(null);

      if (!pointOverlays) return;

      if (viewerType === 'camera') {
        const ann = pointOverlays.camera.annotations.find(
          (a) => a.gcp_id === gcpId,
        );
        const reproj = pointOverlays.camera.reprojected_gcps.find(
          (a) => a.gcp_id === gcpId,
        );
        if (ann && reproj) {
          const midX = (ann.x + reproj.x) / 2;
          const midY = (ann.y + reproj.y) / 2;
          zoomToPoint(cameraViewerRef.current, midX, midY);
        }
      } else {
        const gcp = pointOverlays.map.gcps.find((a) => a.gcp_id === gcpId);
        const proj = pointOverlays.map.projected_annotations.find(
          (a) => a.gcp_id === gcpId,
        );
        if (gcp && proj) {
          const midX = (gcp.x + proj.x) / 2;
          const midY = (gcp.y + proj.y) / 2;
          zoomToPoint(mapViewerRef.current, midX, midY);
        }
      }
    },
    [pointOverlays, zoomToPoint],
  );

  const handleLineClick = useCallback(
    (lineId: string, viewerType: 'camera' | 'map') => {
      setSelectedLineId(lineId);
      setSelectedPointId(null);

      if (!lineOverlaysData) return;

      if (viewerType === 'camera') {
        const ann = lineOverlaysData.camera.annotations.find(
          (l) => l.line_id === lineId,
        );
        if (ann) {
          const midX = (ann.start[0] + ann.end[0]) / 2;
          const midY = (ann.start[1] + ann.end[1]) / 2;
          zoomToPoint(cameraViewerRef.current, midX, midY);
        }
      } else {
        const gcpLine = lineOverlaysData.map.gcp_lines.find(
          (l) => l.line_id === lineId,
        );
        if (gcpLine) {
          const midX = (gcpLine.start[0] + gcpLine.end[0]) / 2;
          const midY = (gcpLine.start[1] + gcpLine.end[1]) / 2;
          zoomToPoint(mapViewerRef.current, midX, midY);
        }
      }
    },
    [lineOverlaysData, zoomToPoint],
  );

  // -----------------------------------------------------------------------
  // Load point test case
  // -----------------------------------------------------------------------

  const loadTestCase = useCallback(
    async (caseId: string) => {
      showStatus('Loading test case...', 'info');

      try {
        // Fetch camera info and map info in parallel
        const [camRes, mapRes] = await Promise.all([
          client.GET('/homography-precision/api/camera-info/', {
            params: {
              query: { case: caseId, tenant_id: selectedTenantId ?? '' },
            },
          }),
          client.GET('/homography-precision/api/map-info/', {
            params: { query: { tenant_id: selectedTenantId ?? '' } },
          }),
        ]);

        if (camRes.error)
          throw new Error(`Camera info failed: ${JSON.stringify(camRes.error)}`);
        if (mapRes.error)
          throw new Error(`Map info failed: ${JSON.stringify(mapRes.error)}`);

        currentCaseRef.current = caseId;
        setCameraInfo(camRes.data as ImageInfo);
        setMapInfo(mapRes.data as ImageInfo);

        // Compute homography
        const homRes = await client.POST(
          '/homography-precision/api/compute-homography/',
          {
            params: { query: { tenant_id: selectedTenantId ?? '' } },
            body: { test_case_name: caseId },
          },
        );

        if (homRes.error)
          throw new Error(
            `Homography failed: ${JSON.stringify(homRes.error)}`,
          );

        const hData = homRes.data!;
        setMetrics((hData.metrics as HomographyMetrics) ?? null);
        setPerPointErrors(
          (hData.per_point_errors as PerPointError[]) ?? [],
        );
        setPointOverlays(
          (hData.overlays as unknown as PointOverlays) ?? null,
        );

        // Clear line state and switch to points mode
        setLineOverlaysData(null);
        setLineMetrics(null);
        setPerLineErrors([]);
        setActiveMode('points');
        setSelectedPointId(null);
        setSelectedLineId(null);

        setStatusType(null);
      } catch (e: unknown) {
        console.error('Failed to load test case:', e);
        showStatus(
          `Failed to load test case: ${e instanceof Error ? e.message : String(e)}`,
          'error',
        );
      }
    },
    [selectedTenantId, showStatus],
  );

  // -----------------------------------------------------------------------
  // Load line test case
  // -----------------------------------------------------------------------

  const loadLineTestCase = useCallback(
    async (caseId: string) => {
      showStatus('Loading line test case...', 'info');

      try {
        // Fetch line test case detail to get point_annotations_ref
        const detailRes = await client.GET(
          '/homography-precision/api/line-test-cases/{name}/',
          {
            params: {
              path: { name: caseId },
              query: { tenant_id: selectedTenantId ?? '' },
            },
          },
        );

        if (detailRes.error)
          throw new Error(
            `Failed to fetch line test case: ${JSON.stringify(detailRes.error)}`,
          );

        const lineTestCase = detailRes.data!;

        // If viewers not yet initialized, determine which image to show
        if (!cameraInfo || !mapInfo) {
          let imageCaseId: string;
          if (!useLineHomography && lineTestCase.point_annotations_ref) {
            imageCaseId = lineTestCase.point_annotations_ref;
          } else if (lineTestCase.point_annotations_ref) {
            imageCaseId = lineTestCase.point_annotations_ref;
          } else {
            throw new Error(
              'Line test case has no point_annotations_ref for viewer initialization',
            );
          }

          currentCaseRef.current = imageCaseId;

          const [camRes, mapRes] = await Promise.all([
            client.GET('/homography-precision/api/camera-info/', {
              params: {
                query: {
                  case: imageCaseId,
                  tenant_id: selectedTenantId ?? '',
                },
              },
            }),
            client.GET('/homography-precision/api/map-info/', {
              params: { query: { tenant_id: selectedTenantId ?? '' } },
            }),
          ]);

          if (camRes.error)
            throw new Error(
              `Camera info failed: ${JSON.stringify(camRes.error)}`,
            );
          if (mapRes.error)
            throw new Error(
              `Map info failed: ${JSON.stringify(mapRes.error)}`,
            );

          setCameraInfo(camRes.data as ImageInfo);
          setMapInfo(mapRes.data as ImageInfo);
        }

        // Compute line errors
        const lineRes = await client.POST(
          '/homography-precision/api/compute-line-errors/',
          {
            params: { query: { tenant_id: selectedTenantId ?? '' } },
            body: {
              test_case_name: caseId,
              use_line_homography: useLineHomography,
            },
          },
        );

        if (lineRes.error) {
          const errObj = lineRes.error as unknown as Record<string, unknown>;
          const errStr =
            typeof errObj === 'object' && 'detail' in errObj
              ? String(errObj.detail)
              : JSON.stringify(errObj);
          throw new Error(errStr);
        }

        const lData = lineRes.data!;
        setLineMetrics((lData.metrics as LineErrorMetrics) ?? null);
        setPerLineErrors(
          (lData.per_line_errors as PerLineError[]) ?? [],
        );
        setLineOverlaysData(
          (lData.line_overlays as unknown as LineOverlays) ?? null,
        );

        // Clear point overlays from viewers and switch to lines mode
        clearOsdOverlays(cameraViewerRef.current, cameraOverlayEls.current);
        clearOsdOverlays(mapViewerRef.current, mapOverlayEls.current);

        setPointOverlays(null);
        setActiveMode('lines');
        setSelectedPointId(null);
        setSelectedLineId(null);

        const source = useLineHomography ? 'line-based' : 'point-based';
        showStatus(`Loaded with ${source} homography`, 'success');
      } catch (e: unknown) {
        console.error('Failed to load line test case:', e);
        showStatus(
          `Failed to load line test case: ${e instanceof Error ? e.message : String(e)}`,
          'error',
        );
      }
    },
    [
      selectedTenantId,
      cameraInfo,
      mapInfo,
      useLineHomography,
      showStatus,
      clearOsdOverlays,
    ],
  );

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  const statusClass =
    statusType === 'error'
      ? styles.statusError
      : statusType === 'success'
        ? styles.statusSuccess
        : statusType === 'info'
          ? styles.statusInfo
          : styles.statusMessage;

  return (
    <div className={styles.pageRoot}>
      {/* --------- top bar --------- */}
      <div className={styles.topBar}>
        <Link to="/" className={styles.backLink}>
          &larr; Back to Tools
        </Link>
        <h1>Homography Precision</h1>

        {/* Point test case selector */}
        <div className={styles.controlGroup}>
          <select
            value={selectedCase}
            onChange={(e) => setSelectedCase(e.target.value)}
          >
            <option value="">-- Select point test case --</option>
            {testCases.map((tc) => (
              <option key={tc.name} value={tc.name}>
                {tc.name} ({tc.annotation_count} annotations)
              </option>
            ))}
          </select>
          <button
            className={styles.btnPrimary}
            disabled={!selectedCase}
            onClick={() => loadTestCase(selectedCase)}
          >
            Load Points
          </button>
        </div>

        {/* Line test case selector */}
        <div className={styles.controlGroup}>
          <select
            value={selectedLineCase}
            onChange={(e) => setSelectedLineCase(e.target.value)}
          >
            <option value="">-- Select line test case --</option>
            {lineTestCases.map((tc) => (
              <option key={tc.name} value={tc.name}>
                {tc.name} ({tc.line_annotation_count} lines)
              </option>
            ))}
          </select>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={useLineHomography}
              onChange={(e) => setUseLineHomography(e.target.checked)}
            />
            Line H
          </label>
          <button
            className={styles.btnPrimary}
            disabled={!selectedLineCase}
            onClick={() => loadLineTestCase(selectedLineCase)}
          >
            Load Lines
          </button>
        </div>

        {/* Status */}
        {statusType && <div className={statusClass}>{statusMsg}</div>}

        {/* Summary metrics (point mode) */}
        <div className={styles.summaryMetrics}>
          <div className={styles.summaryMetric}>
            <div className={styles.metricLabel}>RMSE</div>
            <div
              className={`${styles.metricValue} ${metrics ? getMetricClass(metrics.rmse, { good: 5, medium: 10 }) : ''}`}
            >
              {metrics ? `${metrics.rmse.toFixed(2)} px` : '--'}
            </div>
          </div>
          <div className={styles.summaryMetric}>
            <div className={styles.metricLabel}>Mean</div>
            <div
              className={`${styles.metricValue} ${metrics ? getMetricClass(metrics.mean_reproj_error, { good: 5, medium: 10 }) : ''}`}
            >
              {metrics ? `${metrics.mean_reproj_error.toFixed(2)} px` : '--'}
            </div>
          </div>
          <div className={styles.summaryMetric}>
            <div className={styles.metricLabel}>Max</div>
            <div
              className={`${styles.metricValue} ${metrics ? getMetricClass(metrics.max_reproj_error, { good: 10, medium: 20 }) : ''}`}
            >
              {metrics ? `${metrics.max_reproj_error.toFixed(2)} px` : '--'}
            </div>
          </div>
          <div className={styles.summaryMetric}>
            <div className={styles.metricLabel}>Inliers</div>
            <div
              className={`${styles.metricValue} ${
                metrics
                  ? metrics.num_inliers / metrics.num_gcps >= 0.9
                    ? styles.good
                    : metrics.num_inliers / metrics.num_gcps >= 0.7
                      ? styles.medium
                      : styles.bad
                  : ''
              }`}
            >
              {metrics ? `${metrics.num_inliers}/${metrics.num_gcps}` : '--'}
            </div>
          </div>
        </div>

        {/* Line metrics (only visible in line mode) */}
        {lineMetrics && activeMode === 'lines' && (
          <div className={styles.lineMetricsVisible}>
            <div className={styles.summaryMetric}>
              <div className={styles.metricLabel}>Mean Line Error</div>
              <div
                className={`${styles.metricValue} ${getMetricClass(lineMetrics.mean_line_error, { good: 5, medium: 10 })}`}
              >
                {lineMetrics.mean_line_error.toFixed(2)} px
              </div>
            </div>
            <div className={styles.summaryMetric}>
              <div className={styles.metricLabel}>Max Line Error</div>
              <div
                className={`${styles.metricValue} ${getMetricClass(lineMetrics.max_line_error, { good: 10, medium: 20 })}`}
              >
                {lineMetrics.max_line_error.toFixed(2)} px
              </div>
            </div>
          </div>
        )}
      </div>

      {/* --------- main content: two panels --------- */}
      <div className={styles.mainContent}>
        {/* ---- Camera panel ---- */}
        <div className={styles.panel}>
          <div className={styles.panelTitle}>
            <span>Camera Frame</span>
            <div className={styles.legend}>
              <div className={styles.legendItem}>
                <div className={styles.legendDotBlue} />
                Annotations
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendDotRed} />
                Reprojected
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendDotYellow} />
                Error
              </div>
            </div>
          </div>

          <div className={styles.viewerWrap}>
            {cameraTileSource ? (
              <>
                <OpenSeadragonViewer
                  tileSource={cameraTileSource}
                  onViewerReady={(v) => {
                    cameraViewerRef.current = v;
                  }}
                  showNavigator
                />
                <LineOverlay
                  viewer={cameraViewerRef.current}
                  lines={
                    activeMode === 'points' ? cameraErrorLines : cameraLines
                  }
                  strokeWidth={activeMode === 'points' ? 2 : 3}
                />
              </>
            ) : (
              <div className={styles.loadingPlaceholder}>
                Select a test case to begin
              </div>
            )}
          </div>

          <div className={styles.statsSection}>
            {/* Point errors */}
            {activeMode === 'points' && (
              <div className={styles.tableSection}>
                <PointErrorTable
                  errors={cameraPointTableData}
                  unit="px"
                  viewerType="camera"
                  selectedId={selectedPointId}
                  onRowClick={handlePointClick}
                />
              </div>
            )}

            {/* Line errors */}
            {activeMode === 'lines' && (
              <div className={styles.tableSection}>
                <LineErrorTable
                  errors={cameraLineTableData}
                  unit="px"
                  viewerType="camera"
                  selectedId={selectedLineId}
                  onRowClick={handleLineClick}
                />
              </div>
            )}
          </div>
        </div>

        {/* ---- Map panel ---- */}
        <div className={styles.panel}>
          <div className={styles.panelTitle}>
            <span>GeoTIFF Map</span>
            <div className={styles.legend}>
              <div className={styles.legendItem}>
                <div className={styles.legendDotGreen} />
                GCPs
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendDotRed} />
                Projected
              </div>
              <div className={styles.legendItem}>
                <div className={styles.legendDotYellow} />
                Error
              </div>
            </div>
          </div>

          <div className={styles.viewerWrap}>
            {mapTileSource ? (
              <>
                <OpenSeadragonViewer
                  tileSource={mapTileSource}
                  onViewerReady={(v) => {
                    mapViewerRef.current = v;
                  }}
                  showNavigator
                />
                <LineOverlay
                  viewer={mapViewerRef.current}
                  lines={activeMode === 'points' ? mapErrorLines : mapLines}
                  strokeWidth={activeMode === 'points' ? 2 : 3}
                />
              </>
            ) : (
              <div className={styles.loadingPlaceholder}>
                Select a test case to begin
              </div>
            )}
          </div>

          <div className={styles.statsSection}>
            {/* Point errors (meters) */}
            {activeMode === 'points' && (
              <div className={styles.tableSection}>
                <PointErrorTable
                  errors={mapPointTableData}
                  unit="m"
                  viewerType="map"
                  selectedId={selectedPointId}
                  onRowClick={handlePointClick}
                />
              </div>
            )}

            {/* Line errors (meters) */}
            {activeMode === 'lines' && (
              <div className={styles.tableSection}>
                <LineErrorTable
                  errors={mapLineTableData}
                  unit="m"
                  viewerType="map"
                  selectedId={selectedLineId}
                  onRowClick={handleLineClick}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
