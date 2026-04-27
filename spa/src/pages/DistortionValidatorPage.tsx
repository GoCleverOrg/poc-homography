import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent,
} from 'react';
import client from '../api/client';
import styles from './DistortionValidatorPage.module.css';

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

type ViewMode = 'original' | 'overlay' | 'undistorted';

interface Coefficients {
  k1: number;
  k2: number;
  k3: number;
  p1: number;
  p2: number;
}

interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

interface CalibrationEntry {
  zoom_factor: number;
  validation_rmse?: number;
  reprojection_error_px?: number;
  coefficients: Coefficients;
  intrinsics?: Intrinsics;
  [key: string]: unknown;
}

interface Calibration {
  camera_id: string;
  entries: CalibrationEntry[];
}

interface ComparisonLine {
  points: number[][];
  drawnInMode: 'original' | 'undistorted';
  distortedPoints: number[][];
  undistortedPoints: number[][];
  originalRmse: number;
  originalQuality: string;
  undistortedRmse: number;
  undistortedQuality: string;
}

interface StraightnessResult {
  rmse_pixels: number;
  quality: string;
  is_straight: boolean;
  [key: string]: unknown;
}

interface StatusMessage {
  text: string;
  type: 'info' | 'success' | 'error';
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function getQualityColor(quality: string): string {
  switch (quality) {
    case 'excellent':
      return '#4caf50';
    case 'good':
      return '#8bc34a';
    case 'acceptable':
      return '#ff9800';
    case 'poor':
      return '#f44336';
    default:
      return '#2196f3';
  }
}

function drawLine(
  ctx: CanvasRenderingContext2D,
  points: number[][],
  color: string,
  label: string | null,
  isPreview = false,
) {
  if (points.length === 0) return;

  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    ctx.lineTo(points[i][0], points[i][1]);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = isPreview ? 2 : 3;
  if (isPreview) {
    ctx.setLineDash([5, 5]);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw points
  points.forEach((pt, i) => {
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], i === 0 ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? '#4caf50' : color;
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Draw label
  if (label && points.length >= 2) {
    const midIdx = Math.floor(points.length / 2);
    ctx.font = 'bold 12px system-ui';
    const metrics = ctx.measureText(label);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(
      points[midIdx][0] - metrics.width / 2 - 4,
      points[midIdx][1] - 20,
      metrics.width + 8,
      18,
    );
    ctx.fillStyle = 'white';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, points[midIdx][0], points[midIdx][1] - 11);
  }
}

function statClass(value: number): string {
  if (value < 1) return styles.statGood;
  if (value < 3) return styles.statWarning;
  return styles.statBad;
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function DistortionValidatorPage() {
  /* ---- refs ---- */
  const wrapperRef = useRef<HTMLDivElement>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgOrigRef = useRef<HTMLImageElement>(null);
  const imgUndistRef = useRef<HTMLImageElement>(null);
  const overlaySliderRef = useRef<HTMLInputElement>(null);

  /* ---- state ---- */
  const [cameraIds, setCameraIds] = useState<string[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [selectedZoomIdx, setSelectedZoomIdx] = useState<number | null>(null);
  const [coefficients, setCoefficients] = useState<Coefficients | null>(null);
  const [images, setImages] = useState<{ name: string; source: string }[]>([]);
  const [selectedImage, setSelectedImage] = useState('');

  const [intrinsics, setIntrinsics] = useState<Intrinsics>({
    fx: 1000,
    fy: 1000,
    cx: 960,
    cy: 540,
  });
  const [intrinsicInputs, setIntrinsicInputs] = useState({
    fx: '1000',
    fy: '1000',
    cx: '960',
    cy: '540',
  });
  const [intrinsicsSource, setIntrinsicsSource] = useState('');
  const [intrinsicsSourceColor, setIntrinsicsSourceColor] = useState('#78909c');

  const [viewMode, setViewModeState] = useState<ViewMode>('original');
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [hasUndistortion, setHasUndistortion] = useState(false);
  const [imageWidth, setImageWidth] = useState(0);
  const [imageHeight, setImageHeight] = useState(0);

  const [comparisonLines, setComparisonLines] = useState<ComparisonLine[]>([]);
  const [currentLinePoints, setCurrentLinePoints] = useState<number[][] | null>(
    null,
  );
  const [status, setStatus] = useState<StatusMessage | null>(null);
  const [applying, setApplying] = useState(false);
  const [computingIntrinsics, setComputingIntrinsics] = useState(false);

  const [originalDataUrl, setOriginalDataUrl] = useState<string | null>(null);
  const [undistortedDataUrl, setUndistortedDataUrl] = useState<string | null>(
    null,
  );

  // Panning state stored in ref to avoid re-renders during drag
  const panRef = useRef({
    isPanning: false,
    startX: 0,
    startY: 0,
    scrollStartX: 0,
    scrollStartY: 0,
  });

  /* ---- derived ---- */
  const canApply = coefficients !== null && selectedImage !== '';
  const calEntry =
    calibration && selectedZoomIdx !== null
      ? calibration.entries[selectedZoomIdx]
      : null;

  /* ---- showStatus helper ---- */
  const showStatus = useCallback(
    (text: string, type: 'info' | 'success' | 'error' = 'info') => {
      setStatus({ text, type });
    },
    [],
  );

  /* ================================================================ */
  /* API calls                                                         */
  /* ================================================================ */

  // Load calibration camera IDs
  useEffect(() => {
    (async () => {
      const { data, error } = await client.GET(
        '/distortion-validator/api/calibration-files/',
      );
      if (error) {
        console.error('Failed to load calibration files', error);
        return;
      }
      setCameraIds(data.camera_ids);
    })();
  }, []);

  // Load images
  useEffect(() => {
    (async () => {
      const { data, error } = await client.GET(
        '/distortion-validator/api/images/',
      );
      if (error) {
        console.error('Failed to load images', error);
        return;
      }
      setImages(data.images);
    })();
  }, []);

  // Load calibration when camera selected
  const loadCalibration = useCallback(
    async (cameraId: string) => {
      const { data, error } = await client.POST(
        '/distortion-validator/api/load-calibration/',
        { body: { camera_id: cameraId } },
      );
      if (error) {
        showStatus(`Error loading calibration`, 'error');
        return;
      }

      const cal: Calibration = {
        camera_id: data.camera_id,
        entries: data.entries as unknown as CalibrationEntry[],
      };
      setCalibration(cal);

      // Auto-select if only one entry
      if (cal.entries.length === 1) {
        setSelectedZoomIdx(0);
        selectZoomEntry(cal, 0);
      } else {
        setSelectedZoomIdx(null);
        setCoefficients(null);
      }
    },
    [showStatus],
  );

  const selectZoomEntry = useCallback(
    (cal: Calibration, index: number) => {
      const entry = cal.entries[index];
      if (!entry) return;

      setCoefficients(entry.coefficients);

      if (entry.intrinsics) {
        const intr = entry.intrinsics;
        setIntrinsics(intr);
        setIntrinsicInputs({
          fx: String(intr.fx),
          fy: String(intr.fy),
          cx: String(intr.cx),
          cy: String(intr.cy),
        });
        setIntrinsicsSource('Loaded from calibration file');
        setIntrinsicsSourceColor('#4caf50');
        showStatus(
          `Intrinsics updated: fx=${intr.fx}, fy=${intr.fy}, cx=${intr.cx}, cy=${intr.cy}`,
          'info',
        );
      } else {
        setIntrinsicsSource(
          'Not stored in calibration -- using manual values',
        );
        setIntrinsicsSourceColor('#ff9800');
      }
    },
    [showStatus],
  );

  // Apply undistortion
  const applyUndistortion = useCallback(async () => {
    if (!coefficients || !selectedImage) return;
    setApplying(true);
    showStatus('Applying undistortion...', 'info');

    const { data, error } = await client.POST(
      '/distortion-validator/api/undistort/',
      {
        body: {
          image_path: selectedImage,
          coefficients,
          intrinsics,
          use_opencv: false,
        },
      },
    );

    if (error) {
      showStatus('Error applying undistortion', 'error');
      setApplying(false);
      return;
    }

    const w = data.width;
    const h = data.height;
    setImageWidth(w);
    setImageHeight(h);

    const origUrl = 'data:image/jpeg;base64,' + data.original;
    const undistUrl =
      data.undistorted_url ?? 'data:image/jpeg;base64,' + (data.undistorted ?? '');

    setOriginalDataUrl(origUrl);
    setUndistortedDataUrl(undistUrl);

    // Update default intrinsics if defaults
    setIntrinsicInputs((prev) => {
      const next = { ...prev };
      if (prev.cx === '960') next.cx = String(Math.round(w / 2));
      if (prev.cy === '540') next.cy = String(Math.round(h / 2));
      if (next.cx !== prev.cx || next.cy !== prev.cy) {
        setIntrinsics((old) => ({
          ...old,
          cx: parseFloat(next.cx) || 960,
          cy: parseFloat(next.cy) || 540,
        }));
      }
      return next;
    });

    // Both images are loaded via onLoad handlers on the <img> elements
    // We track load counts via a ref
    loadCountRef.current = 0;
    setApplying(false);
  }, [coefficients, selectedImage, intrinsics, showStatus]);

  const loadCountRef = useRef(0);

  const onImageLoaded = useCallback(() => {
    loadCountRef.current += 1;
    if (loadCountRef.current < 2) return;

    const origImg = imgOrigRef.current;
    if (!origImg) return;
    const w = origImg.naturalWidth;
    const h = origImg.naturalHeight;

    if (containerRef.current) {
      containerRef.current.style.width = w + 'px';
      containerRef.current.style.height = h + 'px';
    }

    setHasUndistortion(true);

    // Setup canvas
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = w;
      canvas.height = h;
    }

    // Fit to wrapper
    const wrapper = wrapperRef.current;
    if (wrapper && w > 0) {
      const newZoom = wrapper.clientWidth / w;
      setZoom(newZoom);
    }

    setViewModeState('overlay');
    setComparisonLines([]);
    setCurrentLinePoints(null);
    setStatus({
      text: 'Ready! Click on the image to draw lines. Use view modes to compare.',
      type: 'success',
    });
  }, []);

  // Measure straightness
  const measureStraightness = useCallback(
    async (
      points: number[][],
      undistort: boolean,
    ): Promise<StraightnessResult> => {
      const body: {
        points: number[][];
        undistort: boolean;
        coefficients?: Coefficients;
        intrinsics?: Intrinsics;
      } = { points, undistort };
      if (undistort) {
        body.coefficients = coefficients!;
        body.intrinsics = intrinsics;
      }
      const resp = await client.POST(
        '/distortion-validator/api/measure-straightness/',
        { body },
      );
      if (resp.error) throw new Error('Failed to measure straightness');
      return resp.data as unknown as StraightnessResult;
    },
    [coefficients, intrinsics],
  );

  // Transform points
  const transformPoints = useCallback(
    async (
      points: number[][],
      direction: 'distort' | 'undistort',
    ): Promise<number[][]> => {
      const { data, error } = await client.POST(
        '/distortion-validator/api/transform-points/',
        {
          body: {
            points,
            direction,
            coefficients: coefficients!,
            intrinsics,
          },
        },
      );
      if (error) throw new Error('Failed to transform points');
      return data.points;
    },
    [coefficients, intrinsics],
  );

  // Compute intrinsics
  const computeIntrinsics = useCallback(async () => {
    setComputingIntrinsics(true);
    const { data, error } = await client.POST(
      '/distortion-validator/api/compute-intrinsics/',
      {
        body: {
          zoom: 1.0,
          image_width: imageWidth || 1920,
          image_height: imageHeight || 1080,
        },
      },
    );
    if (error) {
      showStatus('Failed to compute intrinsics', 'error');
      setComputingIntrinsics(false);
      return;
    }
    const newInputs = {
      fx: data.fx.toFixed(1),
      fy: data.fy.toFixed(1),
      cx: data.cx.toFixed(1),
      cy: data.cy.toFixed(1),
    };
    setIntrinsicInputs(newInputs);
    const newIntr = {
      fx: data.fx,
      fy: data.fy,
      cx: data.cx,
      cy: data.cy,
    };
    setIntrinsics(newIntr);
    setIntrinsicsSource(
      `Computed: f=${data.focal_length_mm.toFixed(1)}mm @ zoom ${data.zoom}x`,
    );
    setIntrinsicsSourceColor('#4caf50');
    showStatus(
      `Intrinsics computed from camera specs: fx=${data.fx.toFixed(1)}`,
      'success',
    );
    setComputingIntrinsics(false);
  }, [imageWidth, imageHeight, showStatus]);

  /* ================================================================ */
  /* View mode management                                              */
  /* ================================================================ */

  const setViewMode = useCallback((mode: ViewMode) => {
    setViewModeState(mode);
  }, []);

  // Update image opacities whenever viewMode, overlayOpacity, or hasUndistortion changes
  useEffect(() => {
    const orig = imgOrigRef.current;
    const undist = imgUndistRef.current;
    if (!orig || !undist) return;

    if (!hasUndistortion) {
      orig.style.opacity = '1';
      undist.style.opacity = '0';
      return;
    }
    switch (viewMode) {
      case 'original':
        orig.style.opacity = '1';
        undist.style.opacity = '0';
        break;
      case 'undistorted':
        orig.style.opacity = '0';
        undist.style.opacity = '1';
        break;
      case 'overlay':
        orig.style.opacity = String(1 - overlayOpacity);
        undist.style.opacity = String(overlayOpacity);
        break;
    }
  }, [viewMode, overlayOpacity, hasUndistortion]);

  /* ================================================================ */
  /* Zoom management                                                   */
  /* ================================================================ */

  const updateZoom = useCallback((newZoom: number) => {
    const clamped = Math.max(0.25, Math.min(5, newZoom));
    setZoom(clamped);
  }, []);

  // Apply zoom transform to DOM
  useEffect(() => {
    const container = containerRef.current;
    const scroller = scrollerRef.current;
    const origImg = imgOrigRef.current;
    if (!container || !scroller || !origImg) return;

    container.style.transform = `scale(${zoom})`;

    const scaledW = origImg.naturalWidth * zoom;
    const scaledH = origImg.naturalHeight * zoom;
    scroller.style.width = scaledW + 'px';
    scroller.style.height = scaledH + 'px';
  }, [zoom]);

  /* ================================================================ */
  /* Canvas rendering                                                  */
  /* ================================================================ */

  const renderLines = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const showOriginalRmse = viewMode === 'original';
    const useUndistortedCoords = viewMode === 'undistorted';

    comparisonLines.forEach((line, i) => {
      const rmse = showOriginalRmse ? line.originalRmse : line.undistortedRmse;
      const quality = showOriginalRmse
        ? line.originalQuality
        : line.undistortedQuality;
      const color = getQualityColor(quality);

      let displayPoints: number[][];
      if (useUndistortedCoords && line.undistortedPoints) {
        displayPoints = line.undistortedPoints;
      } else if (!useUndistortedCoords && line.distortedPoints) {
        displayPoints = line.distortedPoints;
      } else {
        displayPoints = line.points;
      }

      drawLine(
        ctx,
        displayPoints,
        color,
        `L${i + 1}: ${rmse.toFixed(2)}px`,
      );
    });

    // Draw current line being drawn
    if (currentLinePoints && currentLinePoints.length > 0) {
      drawLine(ctx, currentLinePoints, '#ffffff', null, true);
    }
  }, [comparisonLines, currentLinePoints, viewMode]);

  useEffect(() => {
    renderLines();
  }, [renderLines]);

  /* ================================================================ */
  /* Canvas event handlers (click to add point, mousemove for preview) */
  /* ================================================================ */

  const getCanvasCoords = useCallback(
    (e: ReactMouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    },
    [],
  );

  const handleCanvasClick = useCallback(
    (e: ReactMouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e);

      if (!currentLinePoints) {
        setCurrentLinePoints([[coords.x, coords.y]]);
        showStatus('Drawing line. Click more points, press ESC to finish.', 'info');
      } else {
        setCurrentLinePoints((prev) =>
          prev ? [...prev, [coords.x, coords.y]] : [[coords.x, coords.y]],
        );
        showStatus(
          `${currentLinePoints.length + 1} points. Click more or press ESC to finish.`,
          'info',
        );
      }
    },
    [currentLinePoints, getCanvasCoords, showStatus],
  );

  const handleCanvasMouseMove = useCallback(
    (e: ReactMouseEvent<HTMLCanvasElement>) => {
      if (!currentLinePoints || currentLinePoints.length === 0) return;
      const coords = getCanvasCoords(e);
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Re-render existing lines first
      renderLines();

      // Draw preview line from last point to cursor
      const lastPt = currentLinePoints[currentLinePoints.length - 1];
      ctx.beginPath();
      ctx.moveTo(lastPt[0], lastPt[1]);
      ctx.lineTo(coords.x, coords.y);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    },
    [currentLinePoints, getCanvasCoords, renderLines],
  );

  /* ================================================================ */
  /* Finish line (ESC) handler                                         */
  /* ================================================================ */

  const finishLine = useCallback(async () => {
    if (!currentLinePoints || currentLinePoints.length < 2) {
      setCurrentLinePoints(null);
      return;
    }

    showStatus('Measuring line straightness...', 'info');

    try {
      const drawnInMode: 'original' | 'undistorted' =
        viewMode === 'undistorted' ? 'undistorted' : 'original';

      let distortedPoints: number[][];
      let undistortedPoints: number[][];

      if (drawnInMode === 'original') {
        distortedPoints = currentLinePoints;
        undistortedPoints = await transformPoints(
          currentLinePoints,
          'undistort',
        );
      } else {
        undistortedPoints = currentLinePoints;
        distortedPoints = await transformPoints(currentLinePoints, 'distort');
      }

      const [originalResult, undistortedResult] = await Promise.all([
        measureStraightness(distortedPoints, false),
        measureStraightness(distortedPoints, true),
      ]);

      const newLine: ComparisonLine = {
        points: currentLinePoints,
        drawnInMode,
        distortedPoints,
        undistortedPoints,
        originalRmse: originalResult.rmse_pixels,
        originalQuality: originalResult.quality,
        undistortedRmse: undistortedResult.rmse_pixels,
        undistortedQuality: undistortedResult.quality,
      };

      setComparisonLines((prev) => [...prev, newLine]);

      const improvement = (
        ((originalResult.rmse_pixels - undistortedResult.rmse_pixels) /
          originalResult.rmse_pixels) *
        100
      ).toFixed(1);
      showStatus(
        `Line added (drawn in ${drawnInMode} mode)! Original: ${originalResult.rmse_pixels.toFixed(3)}px, Undistorted: ${undistortedResult.rmse_pixels.toFixed(3)}px (${improvement}% improvement)`,
        'success',
      );
    } catch {
      showStatus('Error measuring line', 'error');
    }

    setCurrentLinePoints(null);
  }, [
    currentLinePoints,
    viewMode,
    transformPoints,
    measureStraightness,
    showStatus,
  ]);

  /* ================================================================ */
  /* Keyboard shortcuts                                                */
  /* ================================================================ */

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // ESC to finish line
      if (e.key === 'Escape' && currentLinePoints) {
        e.preventDefault();
        finishLine();
        return;
      }

      // Skip shortcuts when typing in form fields
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        if (e.key === '1') {
          setViewMode('original');
          return;
        }
        if (e.key === '2') {
          setViewMode('overlay');
          return;
        }
        if (e.key === '3') {
          setViewMode('undistorted');
          return;
        }
      }

      // Arrow keys nudge overlay slider
      if (viewMode === 'overlay') {
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          setOverlayOpacity((prev) => Math.max(0, prev - 0.05));
          return;
        }
        if (e.key === 'ArrowRight') {
          e.preventDefault();
          setOverlayOpacity((prev) => Math.min(1, prev + 0.05));
          return;
        }
      }

      // Ctrl+E export
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        handleExport();
      }
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [currentLinePoints, viewMode, finishLine, setViewMode]);

  // Keep overlay slider input in sync with state
  useEffect(() => {
    if (overlaySliderRef.current) {
      overlaySliderRef.current.value = String(Math.round(overlayOpacity * 100));
    }
  }, [overlayOpacity]);

  /* ================================================================ */
  /* Pan & zoom on the image wrapper                                   */
  /* ================================================================ */

  const handleWheel = useCallback(
    (e: WheelEvent<HTMLDivElement>) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const wrapper = wrapperRef.current;
      if (!wrapper) return;

      const oldZoom = zoom;
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newZoom = Math.max(0.25, Math.min(5, oldZoom * delta));
      if (newZoom === oldZoom) return;

      const rect = wrapper.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const imgX = (wrapper.scrollLeft + mouseX) / oldZoom;
      const imgY = (wrapper.scrollTop + mouseY) / oldZoom;

      updateZoom(newZoom);

      // Adjust scroll so the same image point stays under the cursor
      requestAnimationFrame(() => {
        if (wrapper) {
          wrapper.scrollLeft = imgX * newZoom - mouseX;
          wrapper.scrollTop = imgY * newZoom - mouseY;
        }
      });
    },
    [zoom, updateZoom],
  );

  const handleWrapperMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
        e.preventDefault();
        const wrapper = wrapperRef.current;
        if (!wrapper) return;
        panRef.current = {
          isPanning: true,
          startX: e.clientX,
          startY: e.clientY,
          scrollStartX: wrapper.scrollLeft,
          scrollStartY: wrapper.scrollTop,
        };
        wrapper.style.cursor = 'grabbing';
      }
    },
    [],
  );

  useEffect(() => {
    const onMouseMove = (e: globalThis.MouseEvent) => {
      if (!panRef.current.isPanning) return;
      const wrapper = wrapperRef.current;
      if (!wrapper) return;
      const dx = e.clientX - panRef.current.startX;
      const dy = e.clientY - panRef.current.startY;
      wrapper.scrollLeft = panRef.current.scrollStartX - dx;
      wrapper.scrollTop = panRef.current.scrollStartY - dy;
    };

    const onMouseUp = () => {
      if (panRef.current.isPanning) {
        panRef.current.isPanning = false;
        const wrapper = wrapperRef.current;
        if (wrapper) wrapper.style.cursor = '';
      }
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

  /* ================================================================ */
  /* Export                                                             */
  /* ================================================================ */

  const handleExport = useCallback(() => {
    if (!originalDataUrl) return;

    const downloadDataUrl = (dataUrl: string, filename: string) => {
      const a = document.createElement('a');
      a.href = dataUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    };

    const baseName = selectedImage
      ? selectedImage.replace(/\.[^/.]+$/, '')
      : 'image';
    downloadDataUrl(originalDataUrl, baseName + '_original.jpg');

    if (undistortedDataUrl) {
      setTimeout(() => {
        downloadDataUrl(undistortedDataUrl, baseName + '_undistorted.jpg');
      }, 500);
    }

    showStatus('Exporting images...', 'info');
  }, [originalDataUrl, undistortedDataUrl, selectedImage, showStatus]);

  /* ================================================================ */
  /* Update intrinsics from inputs                                     */
  /* ================================================================ */

  const updateIntrinsicsFromInputs = useCallback(() => {
    const newIntr: Intrinsics = {
      fx: parseFloat(intrinsicInputs.fx) || 1000,
      fy: parseFloat(intrinsicInputs.fy) || 1000,
      cx: parseFloat(intrinsicInputs.cx) || 960,
      cy: parseFloat(intrinsicInputs.cy) || 540,
    };
    setIntrinsics(newIntr);
    showStatus(
      `Intrinsics updated: fx=${newIntr.fx}, fy=${newIntr.fy}, cx=${newIntr.cx}, cy=${newIntr.cy}`,
      'info',
    );
  }, [intrinsicInputs, showStatus]);

  /* ================================================================ */
  /* Delete line                                                       */
  /* ================================================================ */

  const deleteLine = useCallback((idx: number) => {
    setComparisonLines((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const clearAllLines = useCallback(() => {
    setComparisonLines([]);
    setCurrentLinePoints(null);
    showStatus('All lines cleared', 'info');
  }, [showStatus]);

  /* ================================================================ */
  /* Summary stats                                                     */
  /* ================================================================ */

  const summaryStats = useMemo(() => {
    if (comparisonLines.length === 0) return null;
    const avgOrig =
      comparisonLines.reduce((s, l) => s + l.originalRmse, 0) /
      comparisonLines.length;
    const avgUndist =
      comparisonLines.reduce((s, l) => s + l.undistortedRmse, 0) /
      comparisonLines.length;
    const avgImprovement =
      avgOrig > 0 ? ((avgOrig - avgUndist) / avgOrig) * 100 : 0;
    return { avgOrig, avgUndist, avgImprovement };
  }, [comparisonLines]);

  /* ================================================================ */
  /* Comparison table                                                  */
  /* ================================================================ */

  const showComparisonTable = useMemo(() => {
    if (!calibration) return false;
    const entries = calibration.entries;
    const hasMultiple = entries.length > 1;
    const hasAnyReproj = entries.some(
      (e) =>
        e.reprojection_error_px !== undefined && e.reprojection_error_px > 0,
    );
    return hasMultiple || hasAnyReproj;
  }, [calibration]);

  /* ================================================================ */
  /* Render                                                            */
  /* ================================================================ */

  const statusClassName = status
    ? `${styles.status} ${status.type === 'info' ? styles.statusInfo : status.type === 'success' ? styles.statusSuccess : styles.statusError}`
    : '';

  return (
    <div className={styles.page}>
      {/* -------- Controls bar -------- */}
      <div className={styles.controlsBar}>
        <div className={styles.controlGroup}>
          <label>Calibration:</label>
          <select
            value={selectedCameraId}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
              const val = e.target.value;
              setSelectedCameraId(val);
              if (val) {
                loadCalibration(val);
              } else {
                setCalibration(null);
                setCoefficients(null);
                setSelectedZoomIdx(null);
              }
            }}
          >
            <option value="">-- Select camera --</option>
            {cameraIds.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.controlGroup}>
          <label>Zoom Entry:</label>
          <select
            disabled={!calibration}
            value={selectedZoomIdx !== null ? String(selectedZoomIdx) : ''}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
              const val = e.target.value;
              if (val !== '' && calibration) {
                const idx = parseInt(val, 10);
                setSelectedZoomIdx(idx);
                selectZoomEntry(calibration, idx);
              }
            }}
          >
            <option value="">-- Select zoom --</option>
            {calibration?.entries.map((entry, i) => {
              const hasReproj =
                entry.reprojection_error_px !== undefined &&
                entry.reprojection_error_px > 0;
              const methodTag = hasReproj ? ' [OpenCV]' : '';
              return (
                <option key={i} value={i}>
                  Zoom {entry.zoom_factor}x (RMSE:{' '}
                  {entry.validation_rmse?.toFixed(3) ?? 'N/A'})
                  {methodTag}
                </option>
              );
            })}
          </select>
        </div>

        <div className={styles.controlGroup}>
          <label>Image:</label>
          <select
            value={selectedImage}
            onChange={(e: ChangeEvent<HTMLSelectElement>) =>
              setSelectedImage(e.target.value)
            }
          >
            <option value="">-- Select image --</option>
            {images.map((img) => (
              <option key={img.name} value={img.name}>
                {img.name}
              </option>
            ))}
          </select>
        </div>

        <button
          className={styles.btnPrimary}
          disabled={!canApply || applying}
          onClick={applyUndistortion}
        >
          {applying ? 'Applying...' : 'Apply Undistortion'}
        </button>
      </div>

      {/* -------- Main content -------- */}
      <div className={styles.mainContent}>
        {/* Unified Image Panel */}
        <div className={styles.imagePanelUnified}>
          <div className={styles.imagePanelHeader}>
            <div className={styles.viewModeGroup}>
              {(['original', 'overlay', 'undistorted'] as const).map((mode) => {
                let activeClass = '';
                if (viewMode === mode) {
                  if (mode === 'original')
                    activeClass = styles.viewModeBtnOriginal;
                  else if (mode === 'undistorted')
                    activeClass = styles.viewModeBtnUndistorted;
                  else activeClass = styles.viewModeBtnActive;
                }
                return (
                  <button
                    key={mode}
                    className={`${styles.viewModeBtn} ${activeClass}`}
                    onClick={() => setViewMode(mode)}
                  >
                    {mode.charAt(0).toUpperCase() + mode.slice(1)}
                  </button>
                );
              })}
            </div>
            <div className={styles.panelControls}>
              <div className={styles.zoomControls}>
                <button onClick={() => updateZoom(zoom / 1.25)}>-</button>
                <span className={styles.zoomDisplay}>
                  {Math.round(zoom * 100)}%
                </span>
                <button onClick={() => updateZoom(zoom * 1.25)}>+</button>
                <button
                  onClick={() => {
                    const natW = imgOrigRef.current?.naturalWidth;
                    if (!natW) {
                      updateZoom(1);
                      return;
                    }
                    const wrapperW = wrapperRef.current?.clientWidth ?? natW;
                    updateZoom(wrapperW / natW);
                  }}
                >
                  Fit
                </button>
              </div>
              <button
                className={`${styles.btnSuccess} ${styles.btnSmall}`}
                disabled={!originalDataUrl}
                onClick={handleExport}
              >
                Export
              </button>
            </div>
          </div>

          <div
            ref={wrapperRef}
            className={styles.imageWrapper}
            onWheel={handleWheel}
            onMouseDown={handleWrapperMouseDown}
          >
            {!hasUndistortion && !originalDataUrl && (
              <div className={styles.emptyState}>
                Select a calibration file and image to begin
              </div>
            )}
            <div
              ref={scrollerRef}
              className={styles.imageScroller}
              style={{
                display:
                  hasUndistortion || originalDataUrl ? 'inline-block' : 'none',
              }}
            >
              <div ref={containerRef} className={styles.imageContainer}>
                <img
                  ref={imgOrigRef}
                  alt=""
                  src={originalDataUrl ?? undefined}
                  onLoad={onImageLoaded}
                />
                <img
                  ref={imgUndistRef}
                  alt=""
                  className={styles.imgStacked}
                  src={undistortedDataUrl ?? undefined}
                  onLoad={onImageLoaded}
                />
                <canvas
                  ref={canvasRef}
                  className={styles.drawCanvas}
                  onClick={handleCanvasClick}
                  onMouseMove={handleCanvasMouseMove}
                />
              </div>
            </div>
          </div>

          {/* Overlay slider */}
          <div
            className={`${styles.overlaySliderBar} ${viewMode !== 'overlay' ? styles.overlaySliderBarHidden : ''}`}
          >
            <label>Original</label>
            <input
              ref={overlaySliderRef}
              type="range"
              className={styles.overlaySlider}
              min="0"
              max="100"
              defaultValue="50"
              onChange={(e) =>
                setOverlayOpacity(parseInt(e.target.value, 10) / 100)
              }
            />
            <span className={styles.sliderValue}>
              Undistorted {Math.round(overlayOpacity * 100)}%
            </span>
          </div>
        </div>

        {/* -------- Sidebar -------- */}
        <div className={styles.sidebar}>
          {/* Calibration Info */}
          {calEntry && (
            <div className={styles.sidebarSection}>
              <h3>Calibration Info</h3>
              <div className={styles.calMethodInfo}>
                <span className={styles.calMethodLabel}>Method: </span>
                <span
                  style={{
                    fontWeight: 600,
                    color:
                      calEntry.reprojection_error_px &&
                      calEntry.reprojection_error_px > 0
                        ? '#2196f3'
                        : '#4caf50',
                  }}
                >
                  {calEntry.reprojection_error_px &&
                  calEntry.reprojection_error_px > 0
                    ? 'OpenCV GCP+Lines'
                    : 'Line Straightness'}
                </span>
              </div>
              <div className={styles.calInfoGrid}>
                <div className={styles.calInfoBox}>
                  <div className={styles.calInfoLabel}>Validation RMSE</div>
                  <div className={styles.calInfoValue}>
                    {calEntry.validation_rmse !== undefined
                      ? calEntry.validation_rmse.toFixed(3) + ' px'
                      : '-'}
                  </div>
                </div>
                <div className={styles.calInfoBox}>
                  <div className={styles.calInfoLabel}>Reproj. Error</div>
                  <div
                    className={styles.calInfoValue}
                    style={{
                      color:
                        calEntry.reprojection_error_px &&
                        calEntry.reprojection_error_px > 0
                          ? '#2196f3'
                          : '#78909c',
                    }}
                  >
                    {calEntry.reprojection_error_px &&
                    calEntry.reprojection_error_px > 0
                      ? calEntry.reprojection_error_px.toFixed(3) + ' px'
                      : 'N/A'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Coefficients */}
          <div className={styles.sidebarSection}>
            <h3>Coefficients</h3>
            <div className={styles.coefficientsGrid}>
              {(['k1', 'k2', 'k3'] as const).map((k) => (
                <div
                  key={k}
                  className={`${styles.coefItem} ${styles.coefRadial}`}
                >
                  <div className={styles.coefName}>{k}</div>
                  <div className={styles.coefValue}>
                    {coefficients ? coefficients[k].toFixed(6) : '-'}
                  </div>
                </div>
              ))}
              {(['p1', 'p2'] as const).map((k) => (
                <div
                  key={k}
                  className={`${styles.coefItem} ${styles.coefTangential}`}
                >
                  <div className={styles.coefName}>{k}</div>
                  <div className={styles.coefValue}>
                    {coefficients ? coefficients[k].toFixed(6) : '-'}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Intrinsics */}
          <div className={styles.sidebarSection}>
            <h3>Intrinsics (Camera Matrix)</h3>
            <div className={styles.intrinsicsWarning}>
              Must match calibration intrinsics!
            </div>
            <div className={styles.intrinsicsGrid}>
              {(
                [
                  ['fx', 'fx (focal X)'],
                  ['fy', 'fy (focal Y)'],
                  ['cx', 'cx (center X)'],
                  ['cy', 'cy (center Y)'],
                ] as const
              ).map(([key, label]) => (
                <div key={key} className={styles.intrinsicsFieldGroup}>
                  <label>{label}</label>
                  <input
                    type="number"
                    value={intrinsicInputs[key]}
                    step={key.startsWith('f') ? '100' : '10'}
                    onChange={(e) =>
                      setIntrinsicInputs((prev) => ({
                        ...prev,
                        [key]: e.target.value,
                      }))
                    }
                  />
                </div>
              ))}
            </div>
            <div className={styles.intrinsicsActions}>
              <button
                className={styles.btnIntrinsicsUpdate}
                onClick={updateIntrinsicsFromInputs}
              >
                Update Intrinsics
              </button>
              <button
                className={styles.btnIntrinsicsCompute}
                disabled={computingIntrinsics}
                onClick={computeIntrinsics}
              >
                {computingIntrinsics ? 'Computing...' : 'Compute from Specs'}
              </button>
            </div>
            {intrinsicsSource && (
              <div
                className={styles.intrinsicsSource}
                style={{ color: intrinsicsSourceColor }}
              >
                {intrinsicsSource}
              </div>
            )}
          </div>

          {/* Method Comparison Table */}
          {showComparisonTable && calibration && (
            <div className={styles.sidebarSection}>
              <h3>Method Comparison</h3>
              <div style={{ overflowX: 'auto' }}>
                <table className={styles.comparisonTable}>
                  <thead>
                    <tr>
                      <th>Zoom</th>
                      <th>Method</th>
                      <th style={{ textAlign: 'right' }}>Reproj. (px)</th>
                      <th style={{ textAlign: 'right' }}>RMSE (px)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calibration.entries.map((entry, i) => {
                      const hasReproj =
                        entry.reprojection_error_px !== undefined &&
                        entry.reprojection_error_px > 0;
                      const method = hasReproj
                        ? 'OpenCV GCP+Lines'
                        : 'Line Straightness';
                      const methodColor = hasReproj ? '#2196f3' : '#4caf50';
                      return (
                        <tr key={i}>
                          <td style={{ color: '#fff' }}>
                            {entry.zoom_factor}x
                          </td>
                          <td
                            style={{ color: methodColor, fontSize: '10px' }}
                          >
                            {method}
                          </td>
                          <td
                            style={{
                              textAlign: 'right',
                              fontFamily: 'monospace',
                              color: hasReproj ? '#2196f3' : '#78909c',
                            }}
                          >
                            {hasReproj
                              ? entry.reprojection_error_px!.toFixed(3)
                              : 'N/A'}
                          </td>
                          <td
                            style={{
                              textAlign: 'right',
                              fontFamily: 'monospace',
                              color: '#fff',
                            }}
                          >
                            {entry.validation_rmse !== undefined
                              ? entry.validation_rmse.toFixed(3)
                              : '-'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Instructions */}
          <div className={styles.sidebarSection}>
            <h3>Instructions</h3>
            <div className={styles.instructions}>
              <strong>1.</strong> Load calibration and select image
              <br />
              <strong>2.</strong> Set intrinsics to match calibration
              <br />
              <strong>3.</strong> Click &quot;Apply Undistortion&quot;
              <br />
              <strong>4.</strong> Use view modes to compare images
              <br />
              <strong>5.</strong> Draw lines, press{' '}
              <span className={styles.key}>ESC</span> to finish
              <br />
              <strong>6.</strong> Compare Original vs Undistorted RMSE
              <br />
              <br />
              <strong>Shortcuts:</strong>
              <br />
              <span className={styles.key}>1</span>{' '}
              <span className={styles.key}>2</span>{' '}
              <span className={styles.key}>3</span> View modes
              <br />
              <span className={styles.key}>&larr;</span>{' '}
              <span className={styles.key}>&rarr;</span> Nudge overlay slider
              <br />
              <span className={styles.key}>Ctrl+E</span> Export images
              <br />
              <span className={styles.key}>Wheel</span> Zoom
              <br />
              <span className={styles.key}>Shift+Drag</span> or{' '}
              <span className={styles.key}>Middle-Click</span> Pan
            </div>
          </div>

          {/* Line Comparisons */}
          <div
            className={`${styles.sidebarSection} ${styles.linesSectionFlex}`}
          >
            <h3>Line Comparisons</h3>
            {status && (
              <div className={statusClassName}>{status.text}</div>
            )}
            <button
              className={`${styles.btnDanger} ${styles.btnSmall}`}
              style={{ marginBottom: 12 }}
              onClick={clearAllLines}
            >
              Clear All Lines
            </button>

            <div className={styles.linesComparison}>
              {comparisonLines.length === 0 ? (
                <div className={styles.emptyLines}>
                  Draw lines on the image to measure straightness
                </div>
              ) : (
                comparisonLines.map((line, i) => {
                  const improvement =
                    line.originalRmse > 0
                      ? ((line.originalRmse - line.undistortedRmse) /
                          line.originalRmse) *
                        100
                      : 0;
                  const improvementClass =
                    improvement > 5
                      ? styles.improvementImproved
                      : improvement < -5
                        ? styles.improvementWorse
                        : styles.improvementSame;
                  const improvementText =
                    improvement > 0
                      ? `${improvement.toFixed(1)}% better`
                      : improvement < 0
                        ? `${Math.abs(improvement).toFixed(1)}% worse`
                        : 'No change';
                  const modeTag =
                    line.drawnInMode === 'undistorted' ? ' [U]' : ' [O]';

                  return (
                    <div key={i} className={styles.lineComparisonItem}>
                      <div className={styles.lineComparisonHeader}>
                        <span
                          className={styles.lineName}
                          title={`Drawn in ${line.drawnInMode} mode`}
                        >
                          Line {i + 1} ({line.points.length} pts){modeTag}
                        </span>
                        <button
                          className={styles.deleteBtn}
                          onClick={() => deleteLine(i)}
                        >
                          &times;
                        </button>
                      </div>
                      <div className={styles.lineComparisonBody}>
                        <div
                          className={`${styles.rmseBox} ${styles.rmseBoxOriginal}`}
                        >
                          <div className={styles.rmseLabel}>Original</div>
                          <div className={styles.rmseValue}>
                            {line.originalRmse.toFixed(3)}
                          </div>
                          <div className={styles.rmseQuality}>
                            {line.originalQuality}
                          </div>
                        </div>
                        <div
                          className={`${styles.rmseBox} ${styles.rmseBoxUndistorted}`}
                        >
                          <div className={styles.rmseLabel}>Undistorted</div>
                          <div className={styles.rmseValue}>
                            {line.undistortedRmse.toFixed(3)}
                          </div>
                          <div className={styles.rmseQuality}>
                            {line.undistortedQuality}
                          </div>
                        </div>
                        <div
                          className={`${styles.improvementIndicator} ${improvementClass}`}
                        >
                          {improvementText}
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {summaryStats && (
              <div className={styles.summaryStats}>
                <div className={styles.statItem}>
                  <div className={styles.statLabel}>Avg Original</div>
                  <div
                    className={`${styles.statValue} ${statClass(summaryStats.avgOrig)}`}
                  >
                    {summaryStats.avgOrig.toFixed(3)}
                  </div>
                </div>
                <div className={styles.statItem}>
                  <div className={styles.statLabel}>Avg Undistorted</div>
                  <div
                    className={`${styles.statValue} ${statClass(summaryStats.avgUndist)}`}
                  >
                    {summaryStats.avgUndist.toFixed(3)}
                  </div>
                </div>
                <div className={styles.statItem}>
                  <div className={styles.statLabel}>Improvement</div>
                  <div
                    className={`${styles.statValue} ${summaryStats.avgImprovement > 10 ? styles.statGood : summaryStats.avgImprovement > 0 ? styles.statWarning : styles.statBad}`}
                  >
                    {summaryStats.avgImprovement.toFixed(1)}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
