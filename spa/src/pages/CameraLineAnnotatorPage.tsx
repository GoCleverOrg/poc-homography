import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from 'react';
import client from '../api/client';
import { useAuth } from '../contexts/AuthContext';
import { useTenant } from '../contexts/TenantContext';
import styles from './CameraLineAnnotatorPage.module.css';

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

interface LineAnnotation {
  line_id: string;
  start_pixel_x: number;
  start_pixel_y: number;
  end_pixel_x: number;
  end_pixel_y: number;
  points?: number[][] | null;
}

interface CameraStatus {
  pan: number | null;
  tilt: number | null;
  zoom: number | null;
}

interface Point {
  x: number;
  y: number;
}

type AnnotationMode = '2point' | 'npoint';

type DragPointType = 'midpoint' | 'start' | 'end' | 'editpoint';

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const NPOINT_MIN_EDGE_PIXELS = 20;
const NPOINT_RECOMMENDED_PIXELS = 30;
const IMAGE_POLL_INTERVAL = 15_000;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function dist(p1: Point, p2: Point): number {
  return Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
}

function generateMidpoints(
  start: Point,
  end: Point,
  maxSegmentLength: number,
): Point[] {
  const lineLength = dist(start, end);
  const numSegments = Math.ceil(lineLength / maxSegmentLength);
  const numMidpoints = numSegments - 1;
  if (numMidpoints <= 0) return [];

  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const pts: Point[] = [];
  for (let i = 1; i <= numMidpoints; i++) {
    const t = i / numSegments;
    pts.push({ x: start.x + dx * t, y: start.y + dy * t });
  }
  return pts;
}

function getQuality(totalPoints: number) {
  if (totalPoints >= NPOINT_RECOMMENDED_PIXELS) {
    return { status: 'good' as const, label: 'Good' };
  }
  if (totalPoints >= NPOINT_MIN_EDGE_PIXELS) {
    return { status: 'marginal' as const, label: 'Marginal' };
  }
  return { status: 'insufficient' as const, label: 'Insufficient' };
}

/* ------------------------------------------------------------------ */
/* Component                                                           */
/* ------------------------------------------------------------------ */

export default function CameraLineAnnotatorPage() {
  const { credentials } = useAuth();
  const { selectedTenantId } = useTenant();

  /* ---- Refs ---- */
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const lineSearchRef = useRef<HTMLInputElement>(null);

  /* ---- Core state ---- */
  const [images, setImages] = useState<string[]>([]);
  const [currentFilename, setCurrentFilename] = useState<string>('');
  const [imageSrc, setImageSrc] = useState<string>('');
  const [imageLoaded, setImageLoaded] = useState(false);
  const [lineIds, setLineIds] = useState<string[]>([]);
  const [mapId, setMapId] = useState<string>('');
  const [annotations, setAnnotations] = useState<LineAnnotation[]>([]);
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>({
    pan: null,
    tilt: null,
    zoom: null,
  });

  /* ---- Mode state ---- */
  const [annotationMode, setAnnotationMode] = useState<AnnotationMode>('2point');
  const [instructionsText, setInstructionsText] = useState(
    'Click twice on the image to create a line annotation. First click = start, second click = end.',
  );

  /* ---- 2-point pending state ---- */
  const [pendingLine, setPendingLine] = useState<{
    startX: number;
    startY: number;
    endX?: number;
    endY?: number;
  } | null>(null);

  /** Mouse-move preview point for the 2-point pending line (ref to avoid re-renders). */
  const previewEndRef = useRef<Point | null>(null);

  /* ---- N-point state ---- */
  const [npointState, setNpointState] = useState<
    'idle' | 'start_placed' | 'adjusting'
  >('idle');
  const [npointStart, setNpointStart] = useState<Point | null>(null);
  const [npointEnd, setNpointEnd] = useState<Point | null>(null);
  const [npointMidpoints, setNpointMidpoints] = useState<Point[]>([]);

  /* ---- Drag state ---- */
  const [isDragging, setIsDragging] = useState(false);
  const [dragPointIndex, setDragPointIndex] = useState(-1);
  const [dragPointType, setDragPointType] = useState<DragPointType>('midpoint');

  /* ---- Edit mode state ---- */
  const [editingIndex, setEditingIndex] = useState(-1);
  const [editPoints, setEditPoints] = useState<number[][]>([]);

  /* ---- Line selection state ---- */
  const [lineInputActive, setLineInputActive] = useState(false);
  const [lineSearchQuery, setLineSearchQuery] = useState('');
  const [selectedLineIndex, setSelectedLineIndex] = useState(-1);

  /* ---- Mode indicator ---- */
  const [modeIndicatorText, setModeIndicatorText] = useState('');
  const [modeIndicatorActive, setModeIndicatorActive] = useState(false);

  /* ---- Coords display ---- */
  const [coordsText, setCoordsText] = useState('Move mouse over image');

  /* ---------------------------------------------------------------- */
  /* Filtered lines                                                    */
  /* ---------------------------------------------------------------- */

  const usedIds = new Set(annotations.map((a) => a.line_id));
  const filteredLines = lineIds
    .filter((id) => id.toLowerCase().includes(lineSearchQuery.toLowerCase()))
    .filter((id) => !usedIds.has(id));

  /* ---------------------------------------------------------------- */
  /* Image loading via authenticated fetch                             */
  /* ---------------------------------------------------------------- */

  const loadImageBlob = useCallback(
    async (filename: string, bust?: number) => {
      if (!selectedTenantId || !filename) return;
      const params = new URLSearchParams({
        tenant_id: selectedTenantId,
        image_filename: filename,
      });
      if (bust) params.set('t', String(bust));
      const url = `/camera-line-annotator/image/?${params}`;

      const headers: Record<string, string> = {};
      if (credentials) {
        headers['Authorization'] = `Basic ${btoa(`${credentials.username}:${credentials.password}`)}`;
      }

      try {
        const resp = await fetch(url, { headers });
        if (!resp.ok) return;
        const blob = await resp.blob();
        const objectUrl = URL.createObjectURL(blob);
        setImageSrc((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return objectUrl;
        });
      } catch {
        // ignore
      }
    },
    [selectedTenantId, credentials],
  );

  // Cleanup blob URL on unmount
  useEffect(() => {
    return () => {
      setImageSrc((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return '';
      });
    };
  }, []);

  /* ---------------------------------------------------------------- */
  /* Canvas drawing                                                    */
  /* ---------------------------------------------------------------- */

  const getMidpointRadius = useCallback(() => {
    if (npointMidpoints.length >= 2) {
      const spacing = dist(npointMidpoints[0], npointMidpoints[1]);
      return Math.max(3, Math.min(7, spacing / 3));
    }
    return 7;
  }, [npointMidpoints]);

  const renderLines = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Helper: draw a simple line with endpoints
    const drawLine = (
      x1: number,
      y1: number,
      x2: number,
      y2: number,
      color: string,
      label: string | null,
      isPending = false,
    ) => {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = color;
      ctx.lineWidth = isPending ? 2 : 3;
      ctx.setLineDash(isPending ? [5, 5] : []);
      ctx.stroke();

      // Start endpoint
      ctx.beginPath();
      ctx.arc(x1, y1, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#4caf50';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      // End endpoint
      ctx.beginPath();
      ctx.arc(x2, y2, 6, 0, Math.PI * 2);
      ctx.fillStyle = isPending ? '#ff9800' : '#f44336';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label at midpoint
      if (label) {
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        ctx.font = 'bold 12px system-ui';
        const metrics = ctx.measureText(label);
        const padding = 4;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(
          midX - metrics.width / 2 - padding,
          midY - 8 - padding,
          metrics.width + padding * 2,
          16 + padding,
        );
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, midX, midY);
      }
    };

    // Helper: draw a polyline
    const drawPolyline = (
      points: number[][],
      color: string,
      label: string | null,
    ) => {
      if (points.length < 2) return;

      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      ctx.stroke();

      // Draw points
      points.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(
          p[0],
          p[1],
          i === 0 ? 6 : i === points.length - 1 ? 6 : 4,
          0,
          Math.PI * 2,
        );
        ctx.fillStyle =
          i === 0 ? '#4caf50' : i === points.length - 1 ? '#f44336' : '#ffeb3b';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Label
      if (label) {
        const midIdx = Math.floor(points.length / 2);
        const midX = points[midIdx][0];
        const midY = points[midIdx][1];
        ctx.font = 'bold 12px system-ui';
        const metrics = ctx.measureText(label);
        const padding = 4;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(
          midX - metrics.width / 2 - padding,
          midY - 8 - padding,
          metrics.width + padding * 2,
          16 + padding,
        );
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, midX, midY);
      }
    };

    // Helper: draw editable polyline
    const drawEditablePolyline = (points: number[][], label: string | null) => {
      if (points.length < 2) return;

      // Ghost straight line
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(150, 150, 150, 0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      ctx.lineTo(points[points.length - 1][0], points[points.length - 1][1]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Polyline
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      ctx.strokeStyle = '#ff9800';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Point handles
      points.forEach((p, i) => {
        const isEndpoint = i === 0 || i === points.length - 1;
        const isBeingDragged = isDragging && dragPointIndex === i;
        const r = isEndpoint ? 8 : isBeingDragged ? 7 : 5;

        ctx.beginPath();
        ctx.arc(p[0], p[1], r, 0, Math.PI * 2);

        if (i === 0) {
          ctx.fillStyle = '#4caf50';
        } else if (i === points.length - 1) {
          ctx.fillStyle = '#f44336';
        } else {
          ctx.fillStyle = isBeingDragged
            ? 'rgba(255, 193, 7, 0.8)'
            : 'rgba(255, 235, 59, 0.6)';
        }
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = isBeingDragged ? 3 : 2;
        ctx.stroke();
      });

      // Label
      if (label) {
        const midIdx = Math.floor(points.length / 2);
        const midX = points[midIdx][0];
        const midY = points[midIdx][1] - 16;
        ctx.font = 'bold 12px system-ui';
        const metrics = ctx.measureText(label);
        const padding = 4;
        ctx.fillStyle = 'rgba(255, 152, 0, 0.9)';
        ctx.fillRect(
          midX - metrics.width / 2 - padding,
          midY - 8 - padding,
          metrics.width + padding * 2,
          16 + padding,
        );
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, midX, midY);
      }
    };

    // Draw completed annotations
    annotations.forEach((ann, i) => {
      if (i === editingIndex) {
        drawEditablePolyline(editPoints, ann.line_id);
      } else if (ann.points && ann.points.length > 1) {
        drawPolyline(ann.points, '#2196f3', ann.line_id);
      } else {
        drawLine(
          ann.start_pixel_x,
          ann.start_pixel_y,
          ann.end_pixel_x,
          ann.end_pixel_y,
          '#2196f3',
          ann.line_id,
        );
      }
    });

    // Draw pending line preview (2-point mode)
    if (annotationMode === '2point' && pendingLine) {
      if (pendingLine.endX !== undefined && pendingLine.endY !== undefined) {
        // Line has been completed (second click done)
        drawLine(
          pendingLine.startX,
          pendingLine.startY,
          pendingLine.endX,
          pendingLine.endY,
          '#ff9800',
          null,
          true,
        );
      } else if (previewEndRef.current) {
        // Mouse-move preview
        drawLine(
          pendingLine.startX,
          pendingLine.startY,
          previewEndRef.current.x,
          previewEndRef.current.y,
          '#ff9800',
          null,
          true,
        );
      } else {
        // Only start marker (first click, no mouse move yet)
        ctx.beginPath();
        ctx.arc(pendingLine.startX, pendingLine.startY, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#ff9800';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Draw N-point mode elements
    if (annotationMode === 'npoint') {
      if (npointState === 'start_placed' && npointStart) {
        ctx.beginPath();
        ctx.arc(npointStart.x, npointStart.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#4caf50';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      if (npointState === 'adjusting' && npointStart && npointEnd) {
        // Ghost line
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(150, 150, 150, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(npointStart.x, npointStart.y);
        ctx.lineTo(npointEnd.x, npointEnd.y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Build all points for polyline
        const allPoints: number[][] = [
          [npointStart.x, npointStart.y],
          ...npointMidpoints.map((p) => [p.x, p.y]),
          [npointEnd.x, npointEnd.y],
        ];

        // Draw polyline
        ctx.beginPath();
        ctx.moveTo(allPoints[0][0], allPoints[0][1]);
        for (let i = 1; i < allPoints.length; i++) {
          ctx.lineTo(allPoints[i][0], allPoints[i][1]);
        }
        ctx.strokeStyle = '#ff9800';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Start point (green)
        ctx.beginPath();
        ctx.arc(npointStart.x, npointStart.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#4caf50';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Midpoints
        const midpointRadius = getMidpointRadius();
        npointMidpoints.forEach((p, i) => {
          const isBeingDragged = isDragging && dragPointIndex === i;
          const r = isBeingDragged ? midpointRadius + 3 : midpointRadius;
          ctx.beginPath();
          ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
          ctx.fillStyle = isBeingDragged
            ? 'rgba(255, 193, 7, 0.25)'
            : 'rgba(255, 235, 59, 0.25)';
          ctx.fill();
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.lineWidth = isBeingDragged ? 2 : 1;
          ctx.stroke();
        });

        // End point (red)
        ctx.beginPath();
        ctx.arc(npointEnd.x, npointEnd.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#f44336';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, [
    annotations,
    editingIndex,
    editPoints,
    annotationMode,
    pendingLine,
    npointState,
    npointStart,
    npointEnd,
    npointMidpoints,
    isDragging,
    dragPointIndex,
    getMidpointRadius,
  ]);

  /* ---------------------------------------------------------------- */
  /* Setup canvas dimensions                                           */
  /* ---------------------------------------------------------------- */

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    renderLines();
  }, [renderLines]);

  /* ---------------------------------------------------------------- */
  /* API calls                                                         */
  /* ---------------------------------------------------------------- */

  const loadImages = useCallback(async () => {
    if (!selectedTenantId) return;
    const { data } = await client.GET('/camera-line-annotator/api/images/', {
      params: { query: { tenant_id: selectedTenantId } },
    });
    if (data) {
      setImages(data);
      if (data.length > 0 && !currentFilename) {
        setCurrentFilename(data[0]);
      }
    }
  }, [selectedTenantId, currentFilename]);

  const loadLineIds = useCallback(async () => {
    if (!selectedTenantId) return;
    const { data } = await client.GET(
      '/camera-line-annotator/api/line-ids/',
      {
        params: { query: { tenant_id: selectedTenantId } },
      },
    );
    if (data) {
      setLineIds(data.line_ids);
      setMapId(data.map_id);
    }
  }, [selectedTenantId]);

  const loadAnnotations = useCallback(
    async (filename: string) => {
      if (!selectedTenantId || !filename) return;
      const { data } = await client.GET(
        '/camera-line-annotator/api/annotations/',
        {
          params: {
            query: { tenant_id: selectedTenantId, image_filename: filename },
          },
        },
      );
      if (data) {
        setAnnotations(data);
      }
    },
    [selectedTenantId],
  );

  const loadCameraStatus = useCallback(
    async (filename: string) => {
      if (!selectedTenantId || !filename) return;
      const { data } = await client.GET(
        '/camera-line-annotator/api/camera-status/',
        {
          params: {
            query: { tenant_id: selectedTenantId, image_filename: filename },
          },
        },
      );
      if (data) {
        setCameraStatus({
          pan: data.pan ?? null,
          tilt: data.tilt ?? null,
          zoom: data.zoom ?? null,
        });
      }
    },
    [selectedTenantId],
  );

  /* ---- Create annotation ---- */
  const createAnnotation = useCallback(
    async (lineId: string, annData: Omit<LineAnnotation, 'line_id'>) => {
      if (!selectedTenantId || !currentFilename) return;
      const { data } = await client.POST(
        '/camera-line-annotator/api/annotations/create/',
        {
          params: { query: { tenant_id: selectedTenantId } },
          body: {
            image_filename: currentFilename,
            line_id: lineId,
            start_pixel_x: annData.start_pixel_x,
            start_pixel_y: annData.start_pixel_y,
            end_pixel_x: annData.end_pixel_x,
            end_pixel_y: annData.end_pixel_y,
            points: annData.points ?? null,
          },
        },
      );
      if (data?.success) {
        const saved = data.annotation;
        // Preserve the local points array if original had one
        if (annData.points) {
          saved.points = annData.points;
        }
        setAnnotations((prev) => [...prev, saved]);
      }
    },
    [selectedTenantId, currentFilename],
  );

  /* ---- Update annotation ---- */
  const updateAnnotation = useCallback(
    async (lineId: string, ann: LineAnnotation) => {
      if (!selectedTenantId || !currentFilename) return;
      await client.PUT(
        '/camera-line-annotator/api/annotations/{line_id}/',
        {
          params: {
            query: { tenant_id: selectedTenantId },
            path: { line_id: lineId },
          },
          body: {
            image_filename: currentFilename,
            start_pixel_x: ann.start_pixel_x,
            start_pixel_y: ann.start_pixel_y,
            end_pixel_x: ann.end_pixel_x,
            end_pixel_y: ann.end_pixel_y,
            points: ann.points ?? null,
          },
        },
      );
    },
    [selectedTenantId, currentFilename],
  );

  /* ---- Delete annotation ---- */
  const deleteAnnotation = useCallback(
    async (lineId: string) => {
      if (!selectedTenantId || !currentFilename) return;
      const { data } = await client.DELETE(
        '/camera-line-annotator/api/annotations/{line_id}/',
        {
          params: {
            query: {
              tenant_id: selectedTenantId,
              image_filename: currentFilename,
            },
            path: { line_id: lineId },
          },
        },
      );
      if (data?.success) {
        setAnnotations((prev) => prev.filter((a) => a.line_id !== lineId));
        if (editingIndex >= 0 && annotations[editingIndex]?.line_id === lineId) {
          setEditingIndex(-1);
          setEditPoints([]);
        }
      }
    },
    [selectedTenantId, currentFilename, editingIndex, annotations],
  );

  /* ---- Rename annotation (delete + create) ---- */
  const renameAnnotation = useCallback(
    async (idx: number, newLineId: string) => {
      if (!selectedTenantId || !currentFilename) return false;
      const ann = annotations[idx];
      if (!ann) return false;

      // Delete old
      const { data: delData } = await client.DELETE(
        '/camera-line-annotator/api/annotations/{line_id}/',
        {
          params: {
            query: {
              tenant_id: selectedTenantId,
              image_filename: currentFilename,
            },
            path: { line_id: ann.line_id },
          },
        },
      );
      if (!delData?.success) return false;

      // Create new
      const { data: createData } = await client.POST(
        '/camera-line-annotator/api/annotations/create/',
        {
          params: { query: { tenant_id: selectedTenantId } },
          body: {
            image_filename: currentFilename,
            line_id: newLineId,
            start_pixel_x: ann.start_pixel_x,
            start_pixel_y: ann.start_pixel_y,
            end_pixel_x: ann.end_pixel_x,
            end_pixel_y: ann.end_pixel_y,
            points: ann.points ?? null,
          },
        },
      );
      if (!createData?.success) return false;

      setAnnotations((prev) =>
        prev.map((a, i) => (i === idx ? { ...a, line_id: newLineId } : a)),
      );
      return true;
    },
    [selectedTenantId, currentFilename, annotations],
  );

  /* ---- Switch image ---- */
  const switchImage = useCallback(
    async (filename: string) => {
      if (!selectedTenantId) return;

      const { data } = await client.POST(
        '/camera-line-annotator/api/switch-image/',
        {
          params: { query: { tenant_id: selectedTenantId } },
          body: { filename },
        },
      );

      if (data?.success) {
        // Reset pending state
        previewEndRef.current = null;
        setPendingLine(null);
        setNpointState('idle');
        setNpointStart(null);
        setNpointEnd(null);
        setNpointMidpoints([]);
        setLineInputActive(false);
        setModeIndicatorActive(false);
        setEditingIndex(-1);
        setEditPoints([]);
        setIsDragging(false);

        setAnnotations(data.annotations);
        setCurrentFilename(data.filename);

        if (data.camera_status) {
          setCameraStatus({
            pan: data.camera_status.pan ?? null,
            tilt: data.camera_status.tilt ?? null,
            zoom: data.camera_status.zoom ?? null,
          });
        }

        // Load the new image
        loadImageBlob(data.filename, Date.now());
      }
    },
    [selectedTenantId, loadImageBlob],
  );

  /* ---------------------------------------------------------------- */
  /* Initial data loading                                              */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    if (!selectedTenantId) return;

    let cancelled = false;

    async function init() {
      await loadImages();
      await loadLineIds();
    }

    if (!cancelled) {
      init();
    }

    return () => {
      cancelled = true;
    };
  }, [selectedTenantId, loadImages, loadLineIds]);

  // Load image, annotations, and camera status when filename changes
  useEffect(() => {
    if (!currentFilename || !selectedTenantId) return;

    loadImageBlob(currentFilename);
    loadAnnotations(currentFilename);
    loadCameraStatus(currentFilename);
  }, [
    currentFilename,
    selectedTenantId,
    loadImageBlob,
    loadAnnotations,
    loadCameraStatus,
  ]);

  // Image polling
  useEffect(() => {
    if (!selectedTenantId) return;

    const timer = setInterval(async () => {
      const { data } = await client.GET(
        '/camera-line-annotator/api/images/',
        {
          params: { query: { tenant_id: selectedTenantId } },
        },
      );
      if (data) {
        setImages((prev) => {
          const existing = new Set(prev);
          const newOnes = data.filter((f: string) => !existing.has(f));
          if (newOnes.length === 0) return prev;
          return [...prev, ...newOnes].sort();
        });
      }
    }, IMAGE_POLL_INTERVAL);

    return () => clearInterval(timer);
  }, [selectedTenantId]);

  /* ---------------------------------------------------------------- */
  /* Re-render lines when relevant state changes                       */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    if (imageLoaded) {
      renderLines();
    }
  }, [imageLoaded, renderLines]);

  /* ---------------------------------------------------------------- */
  /* Image onLoad handler                                              */
  /* ---------------------------------------------------------------- */

  const handleImageLoad = useCallback(() => {
    setImageLoaded(true);
    setupCanvas();
  }, [setupCanvas]);

  /* ---------------------------------------------------------------- */
  /* Mouse coordinate helper                                           */
  /* ---------------------------------------------------------------- */

  const getImageCoords = useCallback(
    (e: ReactMouseEvent | MouseEvent): Point | null => {
      const img = imgRef.current;
      if (!img) return null;
      const rect = img.getBoundingClientRect();
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
      };
    },
    [],
  );

  const getCanvasCoords = useCallback(
    (e: ReactMouseEvent | MouseEvent): Point | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
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

  /* ---------------------------------------------------------------- */
  /* Hit-testing helpers                                                */
  /* ---------------------------------------------------------------- */

  const findNpointAtPosition = useCallback(
    (x: number, y: number) => {
      const endpointThreshold = 12;
      if (npointStart && dist({ x, y }, npointStart) <= endpointThreshold) {
        return { type: 'start' as const, index: -1 };
      }
      if (npointEnd && dist({ x, y }, npointEnd) <= endpointThreshold) {
        return { type: 'end' as const, index: -1 };
      }
      const threshold = getMidpointRadius() + 4;
      for (let i = 0; i < npointMidpoints.length; i++) {
        if (dist({ x, y }, npointMidpoints[i]) <= threshold) {
          return { type: 'midpoint' as const, index: i };
        }
      }
      return null;
    },
    [npointStart, npointEnd, npointMidpoints, getMidpointRadius],
  );

  const findEditPointAtPosition = useCallback(
    (x: number, y: number) => {
      const threshold = 12;
      for (let i = 0; i < editPoints.length; i++) {
        const dx = editPoints[i][0] - x;
        const dy = editPoints[i][1] - y;
        if (Math.sqrt(dx * dx + dy * dy) <= threshold) {
          return i;
        }
      }
      return -1;
    },
    [editPoints],
  );

  /* ---------------------------------------------------------------- */
  /* selectLineId: finalize annotation with chosen line_id             */
  /* ---------------------------------------------------------------- */

  const selectLineId = useCallback(
    async (lineId: string) => {
      if (
        annotationMode === 'npoint' &&
        npointState === 'adjusting' &&
        npointStart &&
        npointEnd
      ) {
        const allPoints: number[][] = [
          [npointStart.x, npointStart.y],
          ...npointMidpoints.map((p) => [p.x, p.y]),
          [npointEnd.x, npointEnd.y],
        ];

        await createAnnotation(lineId, {
          start_pixel_x: npointStart.x,
          start_pixel_y: npointStart.y,
          end_pixel_x: npointEnd.x,
          end_pixel_y: npointEnd.y,
          points: allPoints,
        });
      } else if (pendingLine?.endX !== undefined && pendingLine?.endY !== undefined) {
        await createAnnotation(lineId, {
          start_pixel_x: pendingLine.startX,
          start_pixel_y: pendingLine.startY,
          end_pixel_x: pendingLine.endX,
          end_pixel_y: pendingLine.endY,
        });
      } else {
        return;
      }

      // Reset state
      previewEndRef.current = null;
      setPendingLine(null);
      setNpointState('idle');
      setNpointStart(null);
      setNpointEnd(null);
      setNpointMidpoints([]);
      setIsDragging(false);
      setDragPointIndex(-1);
      setLineInputActive(false);
      setLineSearchQuery('');
      setModeIndicatorActive(false);
    },
    [
      annotationMode,
      npointState,
      npointStart,
      npointEnd,
      npointMidpoints,
      pendingLine,
      createAnnotation,
    ],
  );

  /* ---------------------------------------------------------------- */
  /* Enter / exit edit mode                                            */
  /* ---------------------------------------------------------------- */

  const enterEditMode = useCallback(
    (annIndex: number) => {
      const ann = annotations[annIndex];
      if (!ann) return;

      let pts: number[][];
      if (ann.points && ann.points.length >= 2) {
        pts = ann.points.map((p) => [p[0], p[1]]);
      } else {
        pts = [
          [ann.start_pixel_x, ann.start_pixel_y],
          [ann.end_pixel_x, ann.end_pixel_y],
        ];
      }

      setEditingIndex(annIndex);
      setEditPoints(pts);

      // Cancel any creation in progress
      previewEndRef.current = null;
      setPendingLine(null);
      setNpointState('idle');
      setNpointStart(null);
      setNpointEnd(null);
      setNpointMidpoints([]);
      setLineInputActive(false);
      setIsDragging(false);

      setModeIndicatorText(
        'Editing -- drag points to adjust, press Escape to finish',
      );
      setModeIndicatorActive(true);
    },
    [annotations],
  );

  const exitEditMode = useCallback(async () => {
    if (editingIndex < 0) return;

    const ann = annotations[editingIndex];
    if (!ann) return;

    const startPt = editPoints[0];
    const endPt = editPoints[editPoints.length - 1];

    const updated: LineAnnotation = {
      ...ann,
      start_pixel_x: startPt[0],
      start_pixel_y: startPt[1],
      end_pixel_x: endPt[0],
      end_pixel_y: endPt[1],
      points: editPoints.length > 2 ? editPoints.map((p) => [p[0], p[1]]) : null,
    };

    // Update local state
    setAnnotations((prev) =>
      prev.map((a, i) => (i === editingIndex ? updated : a)),
    );

    // Persist via API
    await updateAnnotation(ann.line_id, {
      ...updated,
      points: editPoints.map((p) => [p[0], p[1]]),
    });

    setEditingIndex(-1);
    setEditPoints([]);
    setIsDragging(false);
    setDragPointIndex(-1);
    setModeIndicatorActive(false);
  }, [editingIndex, annotations, editPoints, updateAnnotation]);

  /* ---------------------------------------------------------------- */
  /* Image click handler                                               */
  /* ---------------------------------------------------------------- */

  const handleImageClick = useCallback(
    (e: ReactMouseEvent<HTMLImageElement>) => {
      if (editingIndex >= 0) return;

      const coords = getImageCoords(e.nativeEvent);
      if (!coords) return;

      if (annotationMode === '2point') {
        if (!pendingLine) {
          setPendingLine({ startX: coords.x, startY: coords.y });
          setModeIndicatorText('Click second point to complete line');
          setModeIndicatorActive(true);
        } else if (pendingLine.endX === undefined) {
          previewEndRef.current = null;
          setPendingLine((prev) =>
            prev ? { ...prev, endX: coords.x, endY: coords.y } : null,
          );
          setModeIndicatorText('Select a line ID from dropdown');
          setLineInputActive(true);
          setLineSearchQuery('');
          setSelectedLineIndex(0);
          requestAnimationFrame(() => lineSearchRef.current?.focus());
        }
      } else if (annotationMode === 'npoint') {
        if (npointState === 'idle') {
          setNpointStart({ x: coords.x, y: coords.y });
          setNpointState('start_placed');
          setModeIndicatorText('Click end point to place midpoints');
          setModeIndicatorActive(true);
        } else if (npointState === 'start_placed' && npointStart) {
          const end = { x: coords.x, y: coords.y };
          setNpointEnd(end);
          setNpointState('adjusting');

          const lineLength = dist(npointStart, end);
          const maxSegmentLength = lineLength / NPOINT_RECOMMENDED_PIXELS;
          setNpointMidpoints(
            generateMidpoints(npointStart, end, maxSegmentLength),
          );

          setModeIndicatorText(
            'Drag points to adjust, press Escape when done',
          );
        }
      }
    },
    [editingIndex, annotationMode, pendingLine, npointState, npointStart, getImageCoords],
  );

  /* ---------------------------------------------------------------- */
  /* Mouse move on image (coords + preview line)                       */
  /* ---------------------------------------------------------------- */

  const handleImageMouseMove = useCallback(
    (e: ReactMouseEvent<HTMLImageElement>) => {
      const coords = getImageCoords(e.nativeEvent);
      if (!coords) return;
      setCoordsText(`Pixel: (${coords.x.toFixed(1)}, ${coords.y.toFixed(1)})`);

      // Update preview line for 2-point mode (use ref to avoid re-render storms)
      if (
        annotationMode === '2point' &&
        pendingLine &&
        pendingLine.endX === undefined
      ) {
        previewEndRef.current = { x: coords.x, y: coords.y };
        renderLines();
      }
    },
    [annotationMode, pendingLine, getImageCoords, renderLines],
  );

  /* ---------------------------------------------------------------- */
  /* Canvas mouse handlers (drag for N-point and edit mode)            */
  /* ---------------------------------------------------------------- */

  const handleCanvasMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLCanvasElement>) => {
      const coords = getCanvasCoords(e.nativeEvent);
      if (!coords) return;

      // Edit mode: drag points
      if (editingIndex >= 0) {
        const hitIdx = findEditPointAtPosition(coords.x, coords.y);
        if (hitIdx >= 0) {
          e.preventDefault();
          e.stopPropagation();
          setIsDragging(true);
          setDragPointIndex(hitIdx);
          setDragPointType('editpoint');
        }
        return;
      }

      // N-point creation mode
      if (annotationMode !== 'npoint' || npointState !== 'adjusting') return;

      const hit = findNpointAtPosition(coords.x, coords.y);
      if (hit) {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
        setDragPointType(hit.type);
        setDragPointIndex(hit.index);
      }
    },
    [
      editingIndex,
      annotationMode,
      npointState,
      getCanvasCoords,
      findEditPointAtPosition,
      findNpointAtPosition,
    ],
  );

  const handleCanvasMouseMove = useCallback(
    (e: ReactMouseEvent<HTMLCanvasElement>) => {
      if (!isDragging) return;
      e.preventDefault();

      const coords = getCanvasCoords(e.nativeEvent);
      if (!coords) return;

      // Edit mode drag - vertical only
      if (dragPointType === 'editpoint' && dragPointIndex >= 0) {
        setEditPoints((prev) =>
          prev.map((p, i) =>
            i === dragPointIndex ? [p[0], coords.y] : p,
          ),
        );
        return;
      }

      // N-point creation drag
      if (dragPointType === 'start' && npointEnd) {
        const newStart = { x: coords.x, y: coords.y };
        setNpointStart(newStart);
        const len = dist(newStart, npointEnd);
        const maxSeg = len / NPOINT_RECOMMENDED_PIXELS;
        setNpointMidpoints(generateMidpoints(newStart, npointEnd, maxSeg));
      } else if (dragPointType === 'end' && npointStart) {
        const newEnd = { x: coords.x, y: coords.y };
        setNpointEnd(newEnd);
        const len = dist(npointStart, newEnd);
        const maxSeg = len / NPOINT_RECOMMENDED_PIXELS;
        setNpointMidpoints(generateMidpoints(npointStart, newEnd, maxSeg));
      } else if (dragPointType === 'midpoint' && dragPointIndex >= 0) {
        setNpointMidpoints((prev) =>
          prev.map((p, i) =>
            i === dragPointIndex ? { x: p.x, y: coords.y } : p,
          ),
        );
      }
    },
    [
      isDragging,
      dragPointType,
      dragPointIndex,
      npointStart,
      npointEnd,
      getCanvasCoords,
    ],
  );

  const handleCanvasMouseUp = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      setDragPointType('midpoint');
      setDragPointIndex(-1);
    }
  }, [isDragging]);

  const handleCanvasMouseLeave = useCallback(() => {
    if (isDragging) {
      setIsDragging(false);
      setDragPointType('midpoint');
      setDragPointIndex(-1);
    }
  }, [isDragging]);

  /* ---------------------------------------------------------------- */
  /* Line search keyboard navigation                                   */
  /* ---------------------------------------------------------------- */

  const handleLineSearchKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedLineIndex((prev) =>
          prev < filteredLines.length - 1 ? prev + 1 : prev,
        );
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedLineIndex((prev) => (prev > 0 ? prev - 1 : prev));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (selectedLineIndex >= 0 && filteredLines[selectedLineIndex]) {
          selectLineId(filteredLines[selectedLineIndex]);
        }
      } else if (e.key === 'Escape') {
        previewEndRef.current = null;
        setPendingLine(null);
        setLineInputActive(false);
        setModeIndicatorActive(false);
      }
    },
    [filteredLines, selectedLineIndex, selectLineId],
  );

  /* ---------------------------------------------------------------- */
  /* Global Escape handler                                             */
  /* ---------------------------------------------------------------- */

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;

      if (editingIndex >= 0) {
        exitEditMode();
        return;
      }

      if (annotationMode === 'npoint') {
        if (npointState === 'adjusting') {
          // Finish N-point: show line ID selection
          setModeIndicatorText('Select a line ID from dropdown');
          setLineInputActive(true);
          setLineSearchQuery('');
          setSelectedLineIndex(0);
          requestAnimationFrame(() => lineSearchRef.current?.focus());
        } else if (npointState === 'start_placed') {
          setNpointState('idle');
          setNpointStart(null);
          setModeIndicatorActive(false);
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [editingIndex, annotationMode, npointState, exitEditMode]);

  /* ---------------------------------------------------------------- */
  /* Quality indicator                                                 */
  /* ---------------------------------------------------------------- */

  const qualityInfo =
    annotationMode === 'npoint' && npointState === 'adjusting'
      ? (() => {
          const totalPoints = 2 + npointMidpoints.length;
          const q = getQuality(totalPoints);
          return { ...q, totalPoints };
        })()
      : null;

  /* ---------------------------------------------------------------- */
  /* Render                                                            */
  /* ---------------------------------------------------------------- */

  // Canvas CSS class
  const canvasClassName = [
    styles.annotationCanvas,
    editingIndex >= 0 || (annotationMode === 'npoint' && npointState === 'adjusting')
      ? styles.draggable
      : '',
    isDragging ? styles.dragging : '',
  ]
    .filter(Boolean)
    .join(' ');

  if (!selectedTenantId) {
    return <div className={styles.loading}>Select a tenant to begin.</div>;
  }

  return (
    <div className={styles.page}>
      {/* Mode indicator */}
      <div
        className={[
          styles.modeIndicator,
          modeIndicatorActive ? styles.modeIndicatorActive : '',
        ]
          .filter(Boolean)
          .join(' ')}
      >
        {modeIndicatorText}
      </div>

      {/* Left panel - Image */}
      <div className={styles.imagePanel}>
        <div className={styles.imageContainer}>
          <div className={styles.imageWrapper}>
            {imageSrc ? (
              <img
                ref={imgRef}
                className={styles.frameImage}
                src={imageSrc}
                alt="Camera Frame"
                onLoad={handleImageLoad}
                onClick={handleImageClick}
                onMouseMove={handleImageMouseMove}
              />
            ) : (
              <div className={styles.loading}>
                {currentFilename ? 'Loading image...' : 'No image selected'}
              </div>
            )}
            <canvas
              ref={canvasRef}
              className={canvasClassName}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onMouseLeave={handleCanvasMouseLeave}
            />
          </div>
        </div>
      </div>

      {/* Right panel - Controls */}
      <div className={styles.controlPanel}>
        {/* Image selector */}
        <div className={styles.panelSection}>
          <h2>Select Image</h2>
          <select
            className={styles.imageSelector}
            value={currentFilename}
            onChange={(e) => switchImage(e.target.value)}
          >
            {images.map((filename) => (
              <option key={filename} value={filename}>
                {filename}
              </option>
            ))}
          </select>
        </div>

        {/* Camera Frame info */}
        <div className={styles.panelSection}>
          <h2>Camera Frame</h2>
          <div className={styles.filename}>{currentFilename}</div>
          <div className={styles.registryInfo}>
            Registry: {mapId ? `(${mapId})` : '(none)'}
          </div>

          {/* Mode selector */}
          <div className={styles.modeSelector}>
            <button
              type="button"
              className={[
                styles.modeBtn,
                annotationMode === '2point' ? styles.modeBtnActive : '',
              ]
                .filter(Boolean)
                .join(' ')}
              onClick={() => {
                if (annotationMode === '2point') return;
                previewEndRef.current = null;
                setPendingLine(null);
                setNpointState('idle');
                setNpointStart(null);
                setNpointEnd(null);
                setNpointMidpoints([]);
                setModeIndicatorActive(false);
                setLineInputActive(false);
                setAnnotationMode('2point');
                setInstructionsText(
                  'Click twice on the image to create a line annotation. First click = start, second click = end.',
                );
              }}
            >
              2-Point
            </button>
            <button
              type="button"
              className={[
                styles.modeBtn,
                annotationMode === 'npoint' ? styles.modeBtnActive : '',
              ]
                .filter(Boolean)
                .join(' ')}
              onClick={() => {
                if (annotationMode === 'npoint') return;
                previewEndRef.current = null;
                setPendingLine(null);
                setNpointState('idle');
                setNpointStart(null);
                setNpointEnd(null);
                setNpointMidpoints([]);
                setModeIndicatorActive(false);
                setLineInputActive(false);
                setAnnotationMode('npoint');
                setInstructionsText(
                  'Click start and end points, then drag midpoints to match line curvature. Press Escape when done.',
                );
              }}
            >
              N-Point
            </button>
          </div>

          <div className={styles.instructions}>{instructionsText}</div>

          {/* Quality indicator */}
          {qualityInfo && (
            <div
              className={[
                styles.qualityIndicator,
                styles.qualityIndicatorActive,
                qualityInfo.status === 'good'
                  ? styles.qualityGood
                  : qualityInfo.status === 'marginal'
                    ? styles.qualityMarginal
                    : styles.qualityInsufficient,
              ]
                .filter(Boolean)
                .join(' ')}
            >
              {qualityInfo.label}: {qualityInfo.totalPoints}/
              {NPOINT_MIN_EDGE_PIXELS} points
            </div>
          )}
        </div>

        {/* Line ID selection */}
        <div
          className={
            lineInputActive
              ? styles.lineInputSectionActive
              : styles.lineInputSection
          }
        >
          <h2>Select Line ID</h2>
          <input
            ref={lineSearchRef}
            type="text"
            className={styles.lineSearch}
            placeholder="Type to filter line IDs..."
            autoComplete="off"
            value={lineSearchQuery}
            onChange={(e) => {
              setLineSearchQuery(e.target.value);
              setSelectedLineIndex(0);
            }}
            onKeyDown={handleLineSearchKeyDown}
          />
          <div className={styles.lineDropdown}>
            {filteredLines.length === 0 ? (
              <div className={styles.lineOptionEmpty}>
                No matching line IDs
              </div>
            ) : (
              filteredLines.map((lineId, i) => (
                <div
                  key={lineId}
                  className={[
                    styles.lineOption,
                    i === selectedLineIndex ? styles.lineOptionSelected : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  onClick={() => selectLineId(lineId)}
                >
                  {lineId}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Annotations list */}
        <div className={styles.panelSection}>
          <h2>
            Annotations (<span>{annotations.length}</span>)
          </h2>
        </div>
        <div className={styles.annotationsList}>
          {annotations.map((ann, i) => {
            const isEditing = i === editingIndex;
            return (
              <div key={ann.line_id} className={styles.annotationItem}>
                <div className={styles.annotationInfo}>
                  <select
                    className={styles.lineIdSelect}
                    value={ann.line_id}
                    onChange={async (e) => {
                      const newId = e.target.value;
                      const ok = await renameAnnotation(i, newId);
                      if (!ok) {
                        e.target.value = ann.line_id;
                      }
                    }}
                  >
                    {lineIds.map((id) => (
                      <option key={id} value={id}>
                        {id}
                      </option>
                    ))}
                  </select>
                  <span className={styles.annotationCoords}>
                    Start: ({ann.start_pixel_x.toFixed(1)},{' '}
                    {ann.start_pixel_y.toFixed(1)}) End: (
                    {ann.end_pixel_x.toFixed(1)},{' '}
                    {ann.end_pixel_y.toFixed(1)})
                  </span>
                </div>
                <div className={styles.annotationActions}>
                  <button
                    type="button"
                    className={[
                      styles.editBtn,
                      isEditing ? styles.editBtnActive : '',
                    ]
                      .filter(Boolean)
                      .join(' ')}
                    onClick={() => {
                      if (editingIndex === i) {
                        exitEditMode();
                      } else {
                        enterEditMode(i);
                      }
                    }}
                  >
                    {isEditing ? 'Editing' : 'Edit'}
                  </button>
                  <button
                    type="button"
                    className={styles.deleteBtn}
                    onClick={() => deleteAnnotation(ann.line_id)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Camera Status */}
        <div className={styles.panelSection}>
          <h2>Camera Status</h2>
          <div className={styles.cameraStatusForm}>
            <div>
              <label htmlFor="cla-pan">Pan</label>
              <input
                id="cla-pan"
                type="number"
                step="0.1"
                placeholder="0.0"
                value={cameraStatus.pan ?? ''}
                readOnly
              />
            </div>
            <div>
              <label htmlFor="cla-tilt">Tilt</label>
              <input
                id="cla-tilt"
                type="number"
                step="0.1"
                placeholder="0.0"
                value={cameraStatus.tilt ?? ''}
                readOnly
              />
            </div>
            <div>
              <label htmlFor="cla-zoom">Zoom</label>
              <input
                id="cla-zoom"
                type="number"
                step="1"
                placeholder="1"
                value={cameraStatus.zoom ?? ''}
                readOnly
              />
            </div>
          </div>
        </div>
      </div>

      {/* Coordinates display */}
      <div className={styles.coordsDisplay}>{coordsText}</div>
    </div>
  );
}
