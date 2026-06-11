import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTenant } from '../contexts/TenantContext';
import client from '../api/client';
import styles from './LensCalibrationPage.module.css';

// ------------------------------------------------------------------ types

interface Intrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

interface DistortionCoefficients {
  k1: number;
  k2: number;
  k3: number;
  p1: number;
  p2: number;
}

interface LineError {
  line_id: string;
  rmse_pixels: number;
  num_samples: number;
}

interface CalibrationResult {
  success: boolean;
  message: string;
  iterations: number;
  initial_error: number;
  final_error: number;
  overall_rmse: number;
  coefficients: DistortionCoefficients;
  intrinsics_used: Intrinsics;
  quality: string;
  line_errors: LineError[] | number[];
  improvement_percent: number;
  optimized_intrinsics?: Intrinsics | null;
}

interface AnnotationRow {
  id: number;
  lineData: Array<{ line_id: string; points: number[][] }>;
  selectedName: string;
}

interface ValidationResult {
  baseline_rmse: number;
  corrected_rmse: number;
  improvement_percent: number;
  num_lines: number;
}

// -- Multi-zoom batch types --
// NOTE: multi-zoom mode uses the annotated line-trace sets as its ONLY input,
// mirroring issue #214's decision that manual lines feed single-zoom only.
type ZoomStatus = 'pending' | 'calibrating' | 'success' | 'failed' | 'skipped';

interface ZoomResult {
  status: ZoomStatus;
  coefficients?: DistortionCoefficients;
  intrinsics?: Intrinsics;
  rmse?: number;
  num_lines?: number;
  message?: string;
  loaded?: boolean; // came from a loaded camera (vs freshly calibrated)
  loadedRmse?: number; // original loaded RMSE, for overwrite confirmation
}

interface PersistedSession {
  multiZoomMode: boolean;
  zoomLevels: number[];
  results: Record<string, ZoomResult>;
  cameraId: string;
}

const ZOOM_STATUS_CLASSES: Record<ZoomStatus, string> = {
  pending: styles.zoomDotPending,
  calibrating: styles.zoomDotCalibrating,
  success: styles.zoomDotSuccess,
  failed: styles.zoomDotFailed,
  skipped: styles.zoomDotSkipped,
};

const LAST_SESSION_KEY = 'lens_calibration_session_last';

function readLastSession(): PersistedSession | null {
  try {
    const raw = localStorage.getItem(LAST_SESSION_KEY);
    if (raw) return JSON.parse(raw) as PersistedSession;
  } catch {
    // ignore malformed / unavailable session
  }
  return null;
}

function parseZoomList(raw: string): number[] {
  const seen = new Set<number>();
  for (const part of raw.split(',')) {
    const n = parseFloat(part.trim());
    if (!Number.isNaN(n) && n > 0) seen.add(n);
  }
  return Array.from(seen).sort((a, b) => a - b);
}

// ------------------------------------------------------------------ helpers

const STATUS_CLASSES: Record<string, string> = {
  info: styles.statusInfo,
  success: styles.statusSuccess,
  error: styles.statusError,
  warning: styles.statusWarning,
};

const QUALITY_CLASSES: Record<string, string> = {
  good: styles.qualityGood,
  acceptable: styles.qualityAcceptable,
  poor: styles.qualityPoor,
};

function qualityFromRmse(rmse: number): string {
  if (rmse < 2.0) return 'good';
  if (rmse < 5.0) return 'acceptable';
  return 'poor';
}

// ------------------------------------------------------------------ component

export default function LensCalibrationPage() {
  const { selectedTenantId } = useTenant();

  // -- Camera Intrinsics --
  const [fx, setFx] = useState(1000);
  const [fy, setFy] = useState(1000);
  const [cx, setCx] = useState(960);
  const [cy, setCy] = useState(540);
  const [cameraId, setCameraId] = useState('my_camera');
  const [zoom, setZoom] = useState(1);
  const [intrinsicsInfo, setIntrinsicsInfo] = useState('');

  // -- Calibration Config --
  const [radialOnly, setRadialOnly] = useState(false);
  const [trainSplitRatio] = useState(0.7);

  // -- Status --
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [statusType, setStatusType] = useState<string>('info');
  const [progressWidth, setProgressWidth] = useState(0);
  const [showProgress, setShowProgress] = useState(false);

  // -- Results --
  const [currentResults, setCurrentResults] = useState<CalibrationResult | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);

  // -- Multi-zoom batch state --
  // Restore the toggle preference from localStorage via a lazy initializer
  // (avoids a setState-in-effect on mount).
  const [multiZoomMode, setMultiZoomMode] = useState<boolean>(() => {
    const parsed = readLastSession();
    return typeof parsed?.multiZoomMode === 'boolean' ? parsed.multiZoomMode : true;
  });
  const [zoomLevels, setZoomLevels] = useState<number[]>([1, 5, 10, 15, 20, 25]);
  const [zoomLevelsInput, setZoomLevelsInput] = useState('1, 5, 10, 15, 20, 25');
  const [addZoomValue, setAddZoomValue] = useState(''); // "Add Zoom Level" (configurator)
  const [addEntryValue, setAddEntryValue] = useState(''); // "Add Zoom Entry" (post-batch)
  const [zoomResults, setZoomResults] = useState<Record<string, ZoomResult>>({});
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; zoom: number } | null>(null);
  // Detect a resumable session at mount-time via a lazy initializer (avoids
  // setState-in-effect). The banner is dismissed/resumed by user action.
  const [resumePrompt, setResumePrompt] = useState<PersistedSession | null>(() => {
    const parsed = readLastSession();
    if (
      parsed &&
      ((parsed.zoomLevels && parsed.zoomLevels.length > 0) ||
        (parsed.results && Object.keys(parsed.results).length > 0))
    ) {
      return parsed;
    }
    return null;
  });
  const sessionLoadedRef = useRef(false);

  // -- Annotation rows --
  const [lineTraceSetNames, setLineTraceSetNames] = useState<string[]>([]);
  const [annotationRows, setAnnotationRows] = useState<AnnotationRow[]>([]);
  const rowCounterRef = useRef(0);

  // -- Load/Save --
  const [cameraIds, setCameraIds] = useState<string[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');

  // -- Loading guards --
  const [computingIntrinsics, setComputingIntrinsics] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loadingCalibration, setLoadingCalibration] = useState(false);
  const [validating, setValidating] = useState(false);

  // ---------------------------------------------------------------- status helpers

  const showStatus = useCallback((message: string, type = 'info') => {
    setStatusMessage(message);
    setStatusType(type);
  }, []);

  // ---------------------------------------------------------------- intrinsics helpers

  const getIntrinsics = useCallback(
    (): Intrinsics => ({
      fx: fx || 1000,
      fy: fy || 1000,
      cx: cx || 960,
      cy: cy || 540,
    }),
    [fx, fy, cx, cy],
  );

  // ---------------------------------------------------------------- compute intrinsics

  const computeIntrinsics = useCallback(async () => {
    setComputingIntrinsics(true);
    try {
      const { data, error } = await client.POST(
        '/lens-calibration/api/compute-intrinsics/',
        {
          body: {
            zoom: zoom || 1.0,
            image_width: Math.round((cx || 960) * 2),
            image_height: Math.round((cy || 540) * 2),
          },
        },
      );
      if (error || !data) {
        showStatus(
          `Error: ${(error as { detail?: string })?.detail ?? 'Failed to compute intrinsics'}`,
          'error',
        );
        return;
      }
      setFx(parseFloat(data.fx.toFixed(1)));
      setFy(parseFloat(data.fy.toFixed(1)));
      setCx(parseFloat(data.cx.toFixed(1)));
      setCy(parseFloat(data.cy.toFixed(1)));
      setIntrinsicsInfo(
        `Computed: f=${data.focal_length_mm.toFixed(1)}mm @ zoom ${data.zoom}x (sensor ${data.sensor_width_mm}mm, base f=${data.base_focal_length_mm}mm)`,
      );
      showStatus(`Intrinsics computed from camera specs: fx=${data.fx.toFixed(1)}`, 'success');
    } catch (e) {
      showStatus(`Failed to compute intrinsics: ${e instanceof Error ? e.message : String(e)}`, 'error');
    }
    setComputingIntrinsics(false);
  }, [zoom, cx, cy, showStatus]);

  // ---------------------------------------------------------------- load line trace sets

  const loadLineTraceSets = useCallback(async () => {
    if (!selectedTenantId) return;
    try {
      const { data } = await client.GET('/lens-calibration/api/line-trace-sets/', {
        params: { query: { tenant_id: selectedTenantId } },
      });
      if (data) {
        setLineTraceSetNames(data.names ?? []);
      }
    } catch (e) {
      console.error('Failed to load line trace sets:', e);
    }
  }, [selectedTenantId]);

  // ---------------------------------------------------------------- fetch line trace set detail

  const fetchLineTraceSet = useCallback(
    async (name: string) => {
      if (!selectedTenantId) return [];
      try {
        const { data } = await client.GET('/lens-calibration/api/line-trace-set-detail/', {
          params: { query: { name, tenant_id: selectedTenantId } },
        });
        if (data) {
          return (data.line_traces ?? []) as Array<{ line_id: string; points: number[][] }>;
        }
      } catch (e) {
        console.error('Failed to fetch line trace set:', e);
      }
      return [];
    },
    [selectedTenantId],
  );

  // ---------------------------------------------------------------- annotation row management

  const addAnnotationRow = useCallback(() => {
    const id = rowCounterRef.current++;
    setAnnotationRows((prev) => [...prev, { id, lineData: [], selectedName: '' }]);
  }, []);

  const removeAnnotationRow = useCallback((rowId: number) => {
    setAnnotationRows((prev) => prev.filter((r) => r.id !== rowId));
  }, []);

  const onAnnotationSelectChange = useCallback(
    async (rowId: number, name: string) => {
      setAnnotationRows((prev) =>
        prev.map((r) => (r.id === rowId ? { ...r, selectedName: name, lineData: [] } : r)),
      );
      if (!name) return;

      const traces = await fetchLineTraceSet(name);
      setAnnotationRows((prev) =>
        prev.map((r) => (r.id === rowId ? { ...r, lineData: traces } : r)),
      );
    },
    [fetchLineTraceSet],
  );

  // ---------------------------------------------------------------- summary computations

  const hasAnnotationLines = annotationRows.some((r) => r.lineData.length > 0);
  const totalAnnotationLines = annotationRows.reduce((sum, r) => sum + r.lineData.length, 0);

  // ---------------------------------------------------------------- run calibration

  const runCalibration = useCallback(async () => {
    setCalibrating(true);
    showStatus('Running annotated lines calibration...', 'info');
    setShowProgress(true);
    setProgressWidth(30);

    try {
      const allLineAnnotations: Array<{ line_id: string; points: number[][] }> = [];
      for (const row of annotationRows) {
        allLineAnnotations.push(...row.lineData);
      }

      if (allLineAnnotations.length === 0) {
        throw new Error('No line annotations loaded');
      }

      setProgressWidth(60);

      const { data, error } = await client.POST(
        '/lens-calibration/api/calibrate-annotated-lines/',
        {
          body: {
            camera_line_annotations: allLineAnnotations,
            intrinsics: {
              fx: fx || 1000,
              fy: fy || 1000,
              cx: cx || 960,
              cy: cy || 540,
              image_width: Math.round((cx || 960) * 2),
              image_height: Math.round((cy || 540) * 2),
            },
            auto_intrinsics: false,
            config: {
              train_split_ratio: trainSplitRatio,
              use_radial_only: radialOnly,
            },
          },
        },
      );

      setProgressWidth(100);

      if (error || !data) {
        throw new Error(
          (error as { detail?: string })?.detail ?? 'Calibration request failed',
        );
      }

      if (data.success) {
        let msg = 'Annotated lines calibration complete!';
        msg += ` | RMSE: ${data.overall_rmse.toFixed(3)} px`;
        if (data.improvement_percent !== undefined) {
          msg += ` | Improvement: ${data.improvement_percent.toFixed(1)}%`;
        }
        showStatus(msg, 'success');

        const result: CalibrationResult = {
          success: data.success,
          message: data.message,
          iterations: data.iterations,
          initial_error: data.initial_error,
          final_error: data.final_error,
          overall_rmse: data.overall_rmse,
          coefficients: data.coefficients,
          intrinsics_used: data.intrinsics_used,
          quality: data.quality,
          line_errors: data.line_errors,
          improvement_percent: data.improvement_percent,
          optimized_intrinsics: data.intrinsics
            ? (data.intrinsics as unknown as Intrinsics)
            : null,
        };
        setCurrentResults(result);

        if (data.intrinsics_used) {
          setIntrinsicsInfo(
            `Used: fx=${data.intrinsics_used.fx.toFixed(1)}, fy=${data.intrinsics_used.fy.toFixed(1)}, cx=${data.intrinsics_used.cx.toFixed(1)}, cy=${data.intrinsics_used.cy.toFixed(1)}`,
          );
        }
      } else {
        showStatus(`Calibration failed: ${data.message}`, 'warning');
      }
    } catch (e) {
      showStatus(`Error: ${e instanceof Error ? e.message : String(e)}`, 'error');
    }

    setCalibrating(false);
    setTimeout(() => {
      setShowProgress(false);
      setProgressWidth(0);
    }, 1000);
  }, [annotationRows, fx, fy, cx, cy, trainSplitRatio, radialOnly, showStatus]);

  // ---------------------------------------------------------------- multi-zoom batch

  // Collect all loaded annotation line traces (the single input method this page has).
  const collectAnnotationLines = useCallback((): Array<{ line_id: string; points: number[][] }> => {
    const all: Array<{ line_id: string; points: number[][] }> = [];
    for (const row of annotationRows) all.push(...row.lineData);
    return all;
  }, [annotationRows]);

  // Calibrate a single zoom: auto-compute intrinsics then calibrate annotated lines.
  // Returns a ZoomResult (success or failed). Pure-ish: does not mutate batch state itself.
  const calibrateSingleZoom = useCallback(
    async (
      zoomValue: number,
      presetLines?: Array<{ line_id: string; points: number[][] }>,
    ): Promise<ZoomResult> => {
      // Reuse already-collected lines when the caller has them (the batch loop
      // collects once for all zooms); otherwise gather them for a one-off run.
      const lines = presetLines ?? collectAnnotationLines();
      if (lines.length === 0) {
        return { status: 'failed', message: 'No annotation lines loaded' };
      }

      try {
        // 1. compute intrinsics for this zoom
        const { data: ci, error: ciErr } = await client.POST(
          '/lens-calibration/api/compute-intrinsics/',
          {
            body: {
              zoom: zoomValue,
              image_width: Math.round((cx || 960) * 2),
              image_height: Math.round((cy || 540) * 2),
            },
          },
        );
        if (ciErr || !ci) {
          return {
            status: 'failed',
            message: (ciErr as { detail?: string })?.detail ?? 'Failed to compute intrinsics',
          };
        }
        const perZoomIntrinsics: Intrinsics = {
          fx: ci.fx,
          fy: ci.fy,
          cx: ci.cx,
          cy: ci.cy,
        };

        // 2. calibrate annotated lines using freshly computed intrinsics
        const { data, error } = await client.POST(
          '/lens-calibration/api/calibrate-annotated-lines/',
          {
            body: {
              camera_line_annotations: lines,
              intrinsics: {
                fx: perZoomIntrinsics.fx,
                fy: perZoomIntrinsics.fy,
                cx: perZoomIntrinsics.cx,
                cy: perZoomIntrinsics.cy,
                image_width: Math.round(perZoomIntrinsics.cx * 2),
                image_height: Math.round(perZoomIntrinsics.cy * 2),
              },
              auto_intrinsics: false,
              config: {
                train_split_ratio: trainSplitRatio,
                use_radial_only: radialOnly,
              },
            },
          },
        );

        if (error || !data) {
          return {
            status: 'failed',
            message: (error as { detail?: string })?.detail ?? 'Calibration request failed',
          };
        }
        if (!data.success) {
          return { status: 'failed', message: data.message || 'Solver reported failure' };
        }

        const numLines = Array.isArray(data.line_errors) ? data.line_errors.length : lines.length;

        return {
          status: 'success',
          coefficients: data.coefficients,
          intrinsics: data.intrinsics_used ?? perZoomIntrinsics,
          rmse: data.overall_rmse,
          num_lines: numLines,
          message: data.message,
        };
      } catch (e) {
        return { status: 'failed', message: e instanceof Error ? e.message : String(e) };
      }
    },
    [collectAnnotationLines, cx, cy, trainSplitRatio, radialOnly],
  );

  // Run the full sequential batch over all configured zoom levels.
  const runMultiZoomCalibration = useCallback(async () => {
    const lines = collectAnnotationLines();
    if (lines.length === 0) {
      showStatus('Load annotation lines before running multi-zoom calibration', 'warning');
      return;
    }
    // Active zoom levels are those not skipped.
    const active = zoomLevels.filter((z) => zoomResults[String(z)]?.status !== 'skipped');
    if (active.length === 0) {
      showStatus('No active zoom levels to calibrate', 'warning');
      return;
    }

    setBatchRunning(true);
    // mark all active zooms pending up front
    setZoomResults((prev) => {
      const next = { ...prev };
      for (const z of active) {
        if (next[String(z)]?.status === 'skipped') continue;
        next[String(z)] = { status: 'pending' };
      }
      return next;
    });

    for (let i = 0; i < active.length; i++) {
      const z = active[i];
      setBatchProgress({ current: i + 1, total: active.length, zoom: z });
      setZoomResults((prev) => ({ ...prev, [String(z)]: { status: 'calibrating' } }));
      showStatus(`Calibrating zoom ${z}x (${i + 1} of ${active.length})`, 'info');

      const result = await calibrateSingleZoom(z, lines);
      setZoomResults((prev) => ({ ...prev, [String(z)]: result }));
    }

    setBatchProgress(null);
    setBatchRunning(false);
    showStatus('Multi-zoom calibration batch complete', 'success');
  }, [collectAnnotationLines, zoomLevels, zoomResults, calibrateSingleZoom, showStatus]);

  // Re-run a single zoom (from progress-table action or "Add Zoom Entry" path).
  const rerunSingleZoom = useCallback(
    async (zoomValue: number) => {
      const lines = collectAnnotationLines();
      if (lines.length === 0) {
        showStatus('Load annotation lines before calibrating', 'warning');
        return;
      }
      setZoomResults((prev) => ({ ...prev, [String(zoomValue)]: { status: 'calibrating' } }));
      showStatus(`Calibrating zoom ${zoomValue}x...`, 'info');
      const result = await calibrateSingleZoom(zoomValue, lines);
      // preserve loadedRmse so overwrite-confirm can compare against the original
      setZoomResults((prev) => {
        const prior = prev[String(zoomValue)];
        return {
          ...prev,
          [String(zoomValue)]: {
            ...result,
            loadedRmse: prior?.loaded ? prior.loadedRmse ?? prior.rmse : prior?.loadedRmse,
          },
        };
      });
      showStatus(
        result.status === 'success'
          ? `Zoom ${zoomValue}x calibrated (RMSE ${result.rmse?.toFixed(3) ?? '---'} px)`
          : `Zoom ${zoomValue}x failed: ${result.message ?? 'unknown'}`,
        result.status === 'success' ? 'success' : 'error',
      );
    },
    [collectAnnotationLines, calibrateSingleZoom, showStatus],
  );

  // Mark a zoom as skipped (removed from the batch).
  const skipZoom = useCallback((zoomValue: number) => {
    setZoomResults((prev) => ({ ...prev, [String(zoomValue)]: { status: 'skipped' } }));
  }, []);

  // ---------------------------------------------------------------- zoom configurator

  const applyZoomLevels = useCallback((levels: number[]) => {
    const sorted = Array.from(new Set(levels.filter((n) => !Number.isNaN(n) && n > 0))).sort(
      (a, b) => a - b,
    );
    setZoomLevels(sorted);
    setZoomLevelsInput(sorted.join(', '));
  }, []);

  // While typing, keep the field bound to the RAW keystrokes (so decimals like
  // "1.5" and unsorted/partial entry remain editable) and only derive the parsed
  // zoom list. Normalization (sort/dedupe/reformat the text) happens on blur.
  const onZoomLevelsInputChange = useCallback((raw: string) => {
    setZoomLevelsInput(raw);
    setZoomLevels(parseZoomList(raw));
  }, []);

  const onZoomLevelsInputBlur = useCallback(() => {
    applyZoomLevels(parseZoomList(zoomLevelsInput));
  }, [applyZoomLevels, zoomLevelsInput]);

  const removeZoomLevel = useCallback(
    (zoomValue: number) => {
      applyZoomLevels(zoomLevels.filter((z) => z !== zoomValue));
      setZoomResults((prev) => {
        const next = { ...prev };
        delete next[String(zoomValue)];
        return next;
      });
    },
    [zoomLevels, applyZoomLevels],
  );

  const addZoomLevel = useCallback(
    (raw: string) => {
      const n = parseFloat(raw);
      if (Number.isNaN(n) || n <= 0) {
        showStatus('Enter a positive zoom value to add', 'warning');
        return;
      }
      applyZoomLevels([...zoomLevels, n]);
      setAddZoomValue('');
    },
    [zoomLevels, applyZoomLevels, showStatus],
  );

  // "Add Zoom Entry": append a new zoom AND immediately calibrate it (single-zoom path).
  const addZoomEntry = useCallback(
    async (raw: string) => {
      const n = parseFloat(raw);
      if (Number.isNaN(n) || n <= 0) {
        showStatus('Enter a positive zoom value to add', 'warning');
        return;
      }
      applyZoomLevels([...zoomLevels, n]);
      setAddEntryValue('');
      await rerunSingleZoom(n);
    },
    [zoomLevels, applyZoomLevels, rerunSingleZoom, showStatus],
  );

  // ---------------------------------------------------------------- validate

  const validateResults = useCallback(async () => {
    if (!currentResults) return;

    setValidating(true);
    showStatus('Validating calibration...', 'info');

    try {
      // Build validation lines from the current annotation data
      const allLines: Array<{
        line_id: string;
        start_x: number;
        start_y: number;
        end_x: number;
        end_y: number;
        pan: number;
        tilt: number;
        zoom: number;
        image_path: string;
        points: number[][];
      }> = [];
      for (const row of annotationRows) {
        for (const trace of row.lineData) {
          if (trace.points.length >= 2) {
            const first = trace.points[0];
            const last = trace.points[trace.points.length - 1];
            allLines.push({
              line_id: trace.line_id,
              start_x: first[0],
              start_y: first[1],
              end_x: last[0],
              end_y: last[1],
              pan: 0,
              tilt: 30,
              zoom: zoom || 1,
              image_path: '',
              points: trace.points,
            });
          }
        }
      }

      if (allLines.length === 0) {
        showStatus('No lines available for validation', 'warning');
        setValidating(false);
        return;
      }

      const { data, error } = await client.POST('/lens-calibration/api/validate/', {
        body: {
          intrinsics: {
            fx: fx || 1000,
            fy: fy || 1000,
            cx: cx || 960,
            cy: cy || 540,
            image_width: Math.round((cx || 960) * 2),
            image_height: Math.round((cy || 540) * 2),
          },
          coefficients: currentResults.coefficients,
          lines: allLines,
        },
      });

      if (error || !data) {
        throw new Error(
          (error as { detail?: string })?.detail ?? 'Validation failed',
        );
      }

      setValidationResult(data);
      showStatus(
        `Validation complete: baseline RMSE ${data.baseline_rmse.toFixed(3)}px -> corrected ${data.corrected_rmse.toFixed(3)}px (${data.improvement_percent.toFixed(1)}% improvement)`,
        'success',
      );
    } catch (e) {
      showStatus(`Validation error: ${e instanceof Error ? e.message : String(e)}`, 'error');
    }

    setValidating(false);
  }, [currentResults, annotationRows, fx, fy, cx, cy, zoom, showStatus]);

  // ---------------------------------------------------------------- save calibration

  const saveCalibration = useCallback(async () => {
    if (!currentResults) return;

    setSaving(true);
    showStatus('Saving calibration...', 'info');

    try {
      const { data, error } = await client.POST('/lens-calibration/api/save/', {
        body: {
          camera_id: cameraId || 'unknown_camera',
          zoom: zoom || 1.0,
          coefficients: currentResults.coefficients,
          intrinsics: getIntrinsics(),
          validation_rmse: currentResults.overall_rmse,
          num_lines: typeof currentResults.line_errors === 'object'
            ? currentResults.line_errors.length
            : 0,
        },
      });

      if (error || !data) {
        throw new Error(
          (error as { detail?: string })?.detail ?? 'Save failed',
        );
      }

      showStatus(`Saved! Camera: ${cameraId}`, 'success');
      loadCalibrationIds();
    } catch (e) {
      showStatus(`Save failed: ${e instanceof Error ? e.message : String(e)}`, 'error');
    }

    setSaving(false);
  }, [currentResults, cameraId, zoom, getIntrinsics, showStatus]);

  // ---------------------------------------------------------------- unified multi-zoom save

  const saveAllZoomEntries = useCallback(async () => {
    // Collect all success entries (freshly calibrated + loaded-and-kept).
    const successZooms = zoomLevels.filter(
      (z) => zoomResults[String(z)]?.status === 'success',
    );
    if (successZooms.length === 0) {
      showStatus('No successful zoom entries to save', 'warning');
      return;
    }

    // Overwrite confirmation: loaded zoom re-calibrated with a different RMSE.
    const conflicts: Array<{ zoom: number; oldRmse: number; newRmse: number }> = [];
    for (const z of successZooms) {
      const r = zoomResults[String(z)];
      if (
        r.loadedRmse !== undefined &&
        r.rmse !== undefined &&
        Math.abs(r.loadedRmse - r.rmse) > 1e-6
      ) {
        conflicts.push({ zoom: z, oldRmse: r.loadedRmse, newRmse: r.rmse });
      }
    }
    if (conflicts.length > 0) {
      const lines = conflicts
        .map(
          (c) =>
            `  zoom ${c.zoom}x: loaded RMSE ${c.oldRmse.toFixed(3)} -> new RMSE ${c.newRmse.toFixed(3)}`,
        )
        .join('\n');
      const ok = window.confirm(
        `The following zoom entries already exist and will be overwritten:\n\n${lines}\n\nProceed with save?`,
      );
      if (!ok) {
        showStatus('Save cancelled', 'warning');
        return;
      }
    }

    setSaving(true);
    showStatus('Saving all zoom entries...', 'info');

    try {
      const zoom_entries = successZooms.map((z) => {
        const r = zoomResults[String(z)];
        return {
          zoom: z,
          coefficients: r.coefficients,
          intrinsics: r.intrinsics ?? null,
          validation_rmse: r.rmse ?? 0,
          num_lines: r.num_lines ?? 0,
        };
      });

      const { data, error } = await client.POST('/lens-calibration/api/save/', {
        body: {
          camera_id: cameraId || 'unknown_camera',
          // Base single-zoom fields are required by the schema; the server uses
          // `zoom_entries` for the multi-zoom batch and ignores these placeholders.
          zoom: zoom_entries[0]?.zoom ?? 1,
          validation_rmse: 0,
          num_lines: 0,
          zoom_entries,
        },
      });

      if (error || !data) {
        throw new Error((error as { detail?: string })?.detail ?? 'Save failed');
      }

      showStatus(
        `Saved ${zoom_entries.length} zoom entries for camera "${cameraId}"`,
        'success',
      );
      loadCalibrationIds();
    } catch (e) {
      showStatus(`Save failed: ${e instanceof Error ? e.message : String(e)}`, 'error');
    }

    setSaving(false);
    // loadCalibrationIds is defined below and stable; mirrors saveCalibration's pattern.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoomLevels, zoomResults, cameraId, showStatus]);

  // ---------------------------------------------------------------- load calibration IDs

  const loadCalibrationIds = useCallback(async () => {
    try {
      const { data } = await client.GET('/lens-calibration/api/calibration-ids/');
      if (data) {
        setCameraIds(data.camera_ids ?? []);
      }
    } catch (e) {
      console.error('Failed to load calibration IDs:', e);
    }
  }, []);

  // ---------------------------------------------------------------- load calibration from repo

  const loadCalibrationFromRepo = useCallback(async () => {
    if (!selectedCameraId) {
      showStatus('Select a camera to load', 'warning');
      return;
    }

    setLoadingCalibration(true);
    showStatus('Loading calibration...', 'info');

    try {
      const { data, error } = await client.POST('/lens-calibration/api/load/', {
        body: { camera_id: selectedCameraId },
      });

      if (error || !data) {
        throw new Error(
          (error as { detail?: string })?.detail ?? 'Load failed',
        );
      }

      setCameraId(data.camera_id);

      const entries = (data.entries ?? []) as Array<Record<string, unknown>>;

      // Loading a camera seeds the session with ALL of its stored entries (marked
      // `loaded`), replacing any prior session state. New zoom levels are then
      // appended via "Add Zoom Entry"; the unified save writes loaded + new together.
      const loadedResults: Record<string, ZoomResult> = {};
      const loadedZooms: number[] = [];
      for (const entry of entries) {
        const zf = (entry.zoom_factor as number) ?? (entry.zoom as number);
        if (zf === undefined || Number.isNaN(zf)) continue;
        const coeffs = entry.coefficients as DistortionCoefficients | undefined;
        const rmse = (entry.validation_rmse as number) ?? 0;
        const numLines = (entry.num_lines_used as number) ?? (entry.num_lines as number) ?? 0;
        const intr = entry.intrinsics as Intrinsics | undefined;
        loadedResults[String(zf)] = {
          status: 'success',
          coefficients: coeffs,
          intrinsics: intr,
          rmse,
          num_lines: numLines,
          message: 'Loaded from repo',
          loaded: true,
          loadedRmse: rmse,
        };
        loadedZooms.push(zf);
      }
      if (loadedZooms.length > 0) {
        applyZoomLevels(loadedZooms);
        setZoomResults(loadedResults);
      }

      if (entries.length > 0) {
        const last = entries[entries.length - 1];
        const zoomFactor = (last.zoom_factor as number) ?? 1;
        setZoom(zoomFactor);

        const coefficients = last.coefficients as DistortionCoefficients | undefined;
        if (coefficients) {
          const validationRmse = (last.validation_rmse as number) ?? 0;
          const numLinesUsed = (last.num_lines_used as number) ?? 0;
          const quality = qualityFromRmse(validationRmse);

          const result: CalibrationResult = {
            success: true,
            message: 'Loaded from repo',
            coefficients,
            overall_rmse: validationRmse,
            quality,
            initial_error: 0,
            final_error: 0,
            improvement_percent: 0,
            iterations: 0,
            line_errors: [],
            intrinsics_used: getIntrinsics(),
            optimized_intrinsics: null,
          };

          // Attach num_lines for display
          (result as CalibrationResult & { num_lines?: number }).num_lines = numLinesUsed;
          setCurrentResults(result);
        }

        const intrinsics = last.intrinsics as Intrinsics | undefined;
        if (intrinsics) {
          setFx(intrinsics.fx);
          setFy(intrinsics.fy);
          setCx(intrinsics.cx);
          setCy(intrinsics.cy);
        }
      }

      showStatus(
        `Loaded! ${entries.length} zoom entries for camera "${data.camera_id}"`,
        'success',
      );
    } catch (e) {
      showStatus(`Load failed: ${e instanceof Error ? e.message : String(e)}`, 'error');
    }

    setLoadingCalibration(false);
  }, [selectedCameraId, getIntrinsics, showStatus, applyZoomLevels]);

  // ---------------------------------------------------------------- copy coefficients

  const copyCoefficients = useCallback(() => {
    if (!currentResults) return;

    const text = `k1: ${currentResults.coefficients.k1}\nk2: ${currentResults.coefficients.k2}\nk3: ${currentResults.coefficients.k3}\np1: ${currentResults.coefficients.p1}\np2: ${currentResults.coefficients.p2}`;

    navigator.clipboard.writeText(text).catch(() => {
      // Fallback: no-op if clipboard unavailable
    });
  }, [currentResults]);

  // ---------------------------------------------------------------- apply optimized intrinsics

  const applyOptimizedIntrinsics = useCallback(() => {
    if (currentResults?.optimized_intrinsics) {
      const oi = currentResults.optimized_intrinsics;
      setFx(parseFloat(oi.fx.toFixed(1)));
      setFy(parseFloat(oi.fy.toFixed(1)));
      setCx(parseFloat(oi.cx.toFixed(1)));
      setCy(parseFloat(oi.cy.toFixed(1)));
      showStatus('Optimized intrinsics applied to input fields', 'success');
    }
  }, [currentResults, showStatus]);

  // ---------------------------------------------------------------- session persistence

  const persistSession = useCallback(() => {
    try {
      const payload: PersistedSession = {
        multiZoomMode,
        zoomLevels,
        results: zoomResults,
        cameraId,
      };
      localStorage.setItem(LAST_SESSION_KEY, JSON.stringify(payload));
    } catch {
      // localStorage unavailable / quota — non-fatal
    }
  }, [multiZoomMode, zoomLevels, zoomResults, cameraId]);

  const applySession = useCallback(
    (s: PersistedSession) => {
      setMultiZoomMode(s.multiZoomMode);
      applyZoomLevels(s.zoomLevels);
      setZoomResults(s.results ?? {});
      if (s.cameraId) setCameraId(s.cameraId);
    },
    [applyZoomLevels],
  );

  const clearSession = useCallback(() => {
    try {
      localStorage.removeItem(LAST_SESSION_KEY);
    } catch {
      // ignore
    }
    setZoomResults({});
    setBatchProgress(null);
    applyZoomLevels([1, 5, 10, 15, 20, 25]);
    setResumePrompt(null);
    showStatus('Session cleared', 'info');
  }, [applyZoomLevels, showStatus]);

  // ---------------------------------------------------------------- effects

  // Load line trace sets + add initial row
  useEffect(() => {
    loadLineTraceSets().then(() => {
      // Add one row by default if empty
      setAnnotationRows((prev) => {
        if (prev.length === 0) {
          const id = rowCounterRef.current++;
          return [{ id, lineData: [], selectedName: '' }];
        }
        return prev;
      });
    });
  }, [loadLineTraceSets]);

  // Load calibration IDs on mount
  useEffect(() => {
    loadCalibrationIds();
  }, [loadCalibrationIds]);

  // Auto-compute intrinsics on mount
  useEffect(() => {
    computeIntrinsics();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-compute intrinsics when zoom changes (if previously computed)
  const prevZoomRef = useRef(zoom);
  useEffect(() => {
    if (prevZoomRef.current !== zoom && intrinsicsInfo.startsWith('Computed:')) {
      computeIntrinsics();
    }
    prevZoomRef.current = zoom;
  }, [zoom, intrinsicsInfo, computeIntrinsics]);

  // Auto-save the session whenever zoom levels, results, or the toggle change.
  // Skip the very first render so we don't immediately re-write the restored
  // session (and so we never overwrite before the user has interacted).
  useEffect(() => {
    if (!sessionLoadedRef.current) {
      sessionLoadedRef.current = true;
      return;
    }
    persistSession();
  }, [zoomLevels, zoomResults, multiZoomMode, persistSession]);

  // ---------------------------------------------------------------- render helpers

  const numLines =
    (currentResults as CalibrationResult & { num_lines?: number } | null)?.num_lines ??
    (Array.isArray(currentResults?.line_errors)
      ? currentResults!.line_errors.length
      : 0);

  // ---------------------------------------------------------------- render

  return (
    <div className={styles.container}>
      {/* Header */}
      <header className={styles.header}>
        <h1 className={styles.title}>Lens Distortion Calibration Tool</h1>
        <Link to="/" className={styles.backLink}>
          &larr; Back to Tools
        </Link>
      </header>

      {/* Resume previous session banner (non-blocking) */}
      {resumePrompt && (
        <div className={styles.resumeBanner}>
          <span>
            Resume previous calibration session
            {resumePrompt.cameraId ? ` for "${resumePrompt.cameraId}"` : ''}?
          </span>
          <div className={styles.resumeBannerActions}>
            <button
              className={`${styles.btnPrimary} ${styles.btnSmall}`}
              type="button"
              onClick={() => {
                applySession(resumePrompt);
                setResumePrompt(null);
                showStatus('Previous session resumed', 'success');
              }}
            >
              Resume
            </button>
            <button
              className={`${styles.btnSecondary} ${styles.btnSmall}`}
              type="button"
              onClick={() => setResumePrompt(null)}
            >
              Discard
            </button>
          </div>
        </div>
      )}

      {/* Multi-Zoom Mode toggle */}
      <div className={styles.modeToggleBar}>
        <label className={styles.checkboxLabel}>
          <input
            type="checkbox"
            checked={multiZoomMode}
            onChange={(e) => setMultiZoomMode(e.target.checked)}
          />
          <strong>Multi-Zoom Mode</strong>
          <span className={styles.intrinsicsInfo}>
            (batch-calibrate multiple zoom levels from the same annotated lines)
          </span>
        </label>
        <button
          className={`${styles.btnSecondary} ${styles.btnSmall}`}
          type="button"
          onClick={clearSession}
        >
          Clear Session
        </button>
      </div>

      <div className={styles.grid}>
        {/* ======================== Camera Intrinsics ======================== */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>Camera Intrinsics</div>
          <div className={styles.cardBody}>
            <div className={styles.formRow}>
              <div className={styles.formGroup}>
                <label className={styles.formLabel} htmlFor="lc-fx">
                  Focal Length X (fx)
                </label>
                <input
                  className={styles.formInput}
                  id="lc-fx"
                  type="number"
                  value={fx}
                  step={0.1}
                  onChange={(e) => setFx(parseFloat(e.target.value) || 0)}
                />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.formLabel} htmlFor="lc-fy">
                  Focal Length Y (fy)
                </label>
                <input
                  className={styles.formInput}
                  id="lc-fy"
                  type="number"
                  value={fy}
                  step={0.1}
                  onChange={(e) => setFy(parseFloat(e.target.value) || 0)}
                />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.formLabel} htmlFor="lc-cx">
                  Principal Point X (cx)
                </label>
                <input
                  className={styles.formInput}
                  id="lc-cx"
                  type="number"
                  value={cx}
                  step={0.1}
                  onChange={(e) => setCx(parseFloat(e.target.value) || 0)}
                />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.formLabel} htmlFor="lc-cy">
                  Principal Point Y (cy)
                </label>
                <input
                  className={styles.formInput}
                  id="lc-cy"
                  type="number"
                  value={cy}
                  step={0.1}
                  onChange={(e) => setCy(parseFloat(e.target.value) || 0)}
                />
              </div>
            </div>

            <div className={styles.formRow2} style={{ marginTop: 12 }}>
              <div className={styles.formGroup}>
                <label className={styles.formLabel} htmlFor="lc-camera-id">
                  Camera ID
                </label>
                <input
                  className={styles.formInput}
                  id="lc-camera-id"
                  type="text"
                  value={cameraId}
                  placeholder="Camera identifier"
                  onChange={(e) => setCameraId(e.target.value)}
                />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.formLabel} htmlFor="lc-zoom">
                  Zoom Factor
                </label>
                <input
                  className={styles.formInput}
                  id="lc-zoom"
                  type="number"
                  value={zoom}
                  step={0.1}
                  min={0.1}
                  onChange={(e) => setZoom(parseFloat(e.target.value) || 1)}
                />
              </div>
            </div>

            <div className={styles.computeRow}>
              <button
                className={styles.btnSecondary}
                type="button"
                disabled={computingIntrinsics}
                onClick={computeIntrinsics}
              >
                {computingIntrinsics ? 'Computing...' : 'Compute from Camera Specs'}
              </button>
              {intrinsicsInfo && (
                <span className={styles.intrinsicsInfo}>{intrinsicsInfo}</span>
              )}
            </div>
          </div>
        </div>

        {/* ======================== Calibration Config ======================== */}
        <div className={styles.card}>
          <div className={styles.cardHeader}>Calibration Configuration</div>
          <div className={styles.cardBody}>
            <div className={styles.formGroup} style={{ marginTop: 12 }}>
              <label className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={radialOnly}
                  onChange={(e) => setRadialOnly(e.target.checked)}
                />
                Radial distortion only (k1, k2, k3) - ignore tangential (p1, p2)
              </label>
            </div>
          </div>
        </div>

        {/* ======================== Input Lines ======================== */}
        <div className={`${styles.card} ${styles.cardFullwidth}`}>
          <div className={styles.cardHeader}>Input Lines</div>
          <div className={styles.cardBody}>
            <h4 className={styles.sectionTitle}>Line Annotation Files</h4>

            {/* Annotation rows */}
            {annotationRows.map((row, idx) => (
              <div key={row.id} className={styles.annotationRow}>
                <div className={styles.annotationRowHeader}>
                  <strong className={styles.annotationRowTitle}>
                    Annotations {idx + 1}
                  </strong>
                  <button
                    className={`${styles.btnSecondary} ${styles.btnSmall}`}
                    type="button"
                    onClick={() => removeAnnotationRow(row.id)}
                  >
                    Remove
                  </button>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Line Trace Set</label>
                  <select
                    className={styles.formSelect}
                    value={row.selectedName}
                    onChange={(e) => onAnnotationSelectChange(row.id, e.target.value)}
                  >
                    <option value="">-- Select line trace set --</option>
                    {lineTraceSetNames.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className={styles.annotationRowInfo}>
                  {row.lineData.length} line traces
                </div>
              </div>
            ))}

            <button
              className={styles.btnSecondary}
              style={{ marginTop: 8 }}
              type="button"
              onClick={addAnnotationRow}
            >
              + Add Line Annotations
            </button>

            {hasAnnotationLines && (
              <div className={styles.statusInfo} style={{ marginTop: 12 }}>
                <strong>Ready:</strong> {totalAnnotationLines} line annotation(s) loaded
              </div>
            )}

            {/* Single-zoom run button only when multi-zoom mode is OFF (fallback path). */}
            {!multiZoomMode && (
              <button
                className={styles.btnPrimary}
                style={{ marginTop: 12 }}
                type="button"
                disabled={!hasAnnotationLines || calibrating}
                onClick={runCalibration}
              >
                {calibrating ? 'Running...' : 'Run Annotated Lines Calibration'}
              </button>
            )}

            <p className={styles.helpText}>
              Load line annotation files with N-point traces of lines that should be
              straight.
              <br />
              The solver optimises distortion coefficients to minimise line curvature
              after undistortion.
            </p>
          </div>
        </div>

        {/* ======================== Multi-Zoom Batch Wizard ======================== */}
        {multiZoomMode && (
          <div className={`${styles.card} ${styles.cardFullwidth}`}>
            <div className={styles.cardHeader}>Multi-Zoom Batch Calibration</div>
            <div className={styles.cardBody}>
              {/* --- Zoom configurator --- */}
              <h4 className={styles.sectionTitle}>Zoom Levels</h4>
              <div className={styles.formGroup}>
                <label className={styles.formLabel} htmlFor="lc-zoom-levels">
                  Comma-separated zoom values
                </label>
                <input
                  className={styles.formInput}
                  id="lc-zoom-levels"
                  type="text"
                  value={zoomLevelsInput}
                  placeholder="1, 5, 10, 15, 20, 25"
                  onChange={(e) => onZoomLevelsInputChange(e.target.value)}
                  onBlur={onZoomLevelsInputBlur}
                />
              </div>

              {/* Chips */}
              <div className={styles.zoomChips}>
                {zoomLevels.length === 0 && (
                  <span className={styles.intrinsicsInfo}>No zoom levels configured</span>
                )}
                {zoomLevels.map((z) => (
                  <span key={z} className={styles.zoomChip}>
                    {z}x
                    <button
                      type="button"
                      className={styles.zoomChipRemove}
                      aria-label={`Remove zoom ${z}`}
                      onClick={() => removeZoomLevel(z)}
                    >
                      &times;
                    </button>
                  </span>
                ))}
              </div>

              {/* Add zoom level + presets */}
              <div className={styles.zoomConfigRow}>
                <input
                  className={styles.zoomAddInput}
                  type="number"
                  step={0.1}
                  min={0.1}
                  value={addZoomValue}
                  placeholder="e.g. 30"
                  onChange={(e) => setAddZoomValue(e.target.value)}
                />
                <button
                  className={`${styles.btnSecondary} ${styles.btnSmall}`}
                  type="button"
                  onClick={() => addZoomLevel(addZoomValue)}
                >
                  Add Zoom Level
                </button>
                <button
                  className={`${styles.btnSecondary} ${styles.btnSmall}`}
                  type="button"
                  onClick={() => applyZoomLevels([1, 5, 10, 15, 20, 25])}
                >
                  1-25x Standard
                </button>
                <button
                  className={`${styles.btnSecondary} ${styles.btnSmall}`}
                  type="button"
                  onClick={() => applyZoomLevels([1, 5, 10, 15, 20, 25, 30, 35, 40])}
                >
                  1-40x Extended
                </button>
              </div>

              {/* --- Run batch --- */}
              <div className={styles.actions} style={{ marginTop: 16 }}>
                <button
                  className={styles.btnPrimary}
                  type="button"
                  disabled={!hasAnnotationLines || batchRunning || zoomLevels.length === 0}
                  onClick={runMultiZoomCalibration}
                >
                  {batchRunning ? 'Calibrating...' : 'Run Multi-Zoom Calibration'}
                </button>
                <div className={styles.zoomAddEntryRow}>
                  <input
                    className={styles.zoomAddInput}
                    type="number"
                    step={0.1}
                    min={0.1}
                    value={addEntryValue}
                    placeholder="new zoom"
                    onChange={(e) => setAddEntryValue(e.target.value)}
                  />
                  <button
                    className={styles.btnSecondary}
                    type="button"
                    disabled={!hasAnnotationLines || batchRunning}
                    onClick={() => addZoomEntry(addEntryValue)}
                  >
                    Add Zoom Entry
                  </button>
                </div>
              </div>

              {!hasAnnotationLines && (
                <div className={styles.statusWarning} style={{ marginTop: 12 }}>
                  Load annotation lines (above) to enable multi-zoom calibration.
                </div>
              )}

              {/* --- Progress header --- */}
              {batchProgress && (
                <div className={styles.statusInfo} style={{ marginTop: 12 }}>
                  Calibrating zoom {batchProgress.zoom}x ({batchProgress.current} of{' '}
                  {batchProgress.total})
                </div>
              )}

              {/* --- Progress table --- */}
              {zoomLevels.length > 0 && (
                <div className={styles.linesList} style={{ marginTop: 12, maxHeight: 'none' }}>
                  <table className={styles.linesTable}>
                    <thead>
                      <tr>
                        <th>Zoom</th>
                        <th>Status</th>
                        <th>RMSE</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {zoomLevels.map((z) => {
                        const r = zoomResults[String(z)] ?? { status: 'pending' as ZoomStatus };
                        const quality =
                          r.status === 'success' && r.rmse !== undefined
                            ? qualityFromRmse(r.rmse)
                            : null;
                        return (
                          <tr key={z}>
                            <td className={styles.mono}>{z}x</td>
                            <td>
                              <span
                                className={`${styles.zoomDot} ${ZOOM_STATUS_CLASSES[r.status]}`}
                              />
                              {r.status}
                              {r.status === 'failed' && r.message ? (
                                <span className={styles.zoomFailMsg}> — {r.message}</span>
                              ) : null}
                            </td>
                            <td className={styles.mono}>
                              {r.status === 'success' && r.rmse !== undefined ? (
                                <>
                                  {r.rmse.toFixed(3)}{' '}
                                  {quality && (
                                    <span className={QUALITY_CLASSES[quality] ?? styles.qualityBadge}>
                                      {quality}
                                    </span>
                                  )}
                                </>
                              ) : (
                                '—'
                              )}
                            </td>
                            <td>
                              {(r.status === 'success' || r.status === 'failed') && (
                                <button
                                  className={`${styles.btnSecondary} ${styles.btnSmall}`}
                                  type="button"
                                  disabled={batchRunning || !hasAnnotationLines}
                                  onClick={() => rerunSingleZoom(z)}
                                >
                                  Re-run
                                </button>
                              )}
                              {(r.status === 'pending' || r.status === 'failed') && (
                                <button
                                  className={`${styles.btnSecondary} ${styles.btnSmall}`}
                                  style={{ marginLeft: 6 }}
                                  type="button"
                                  disabled={batchRunning}
                                  onClick={() => skipZoom(z)}
                                >
                                  Skip
                                </button>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {/* --- Unified save --- */}
              <div className={styles.actions} style={{ marginTop: 16 }}>
                <button
                  className={styles.btnSuccess}
                  type="button"
                  disabled={saving || batchRunning}
                  onClick={saveAllZoomEntries}
                >
                  {saving ? 'Saving...' : 'Save All Zoom Entries'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ======================== Calibration Status ======================== */}
        <div className={`${styles.card} ${styles.cardFullwidth}`}>
          <div className={styles.cardHeader}>Calibration</div>
          <div className={styles.cardBody}>
            {statusMessage && (
              <div className={STATUS_CLASSES[statusType] ?? styles.statusInfo}>
                {statusMessage}
              </div>
            )}

            {showProgress && (
              <div className={styles.progressBar}>
                <div
                  className={styles.progressFill}
                  style={{ width: `${progressWidth}%` }}
                />
              </div>
            )}

            <div className={styles.actions}>
              <button
                className={styles.btnSecondary}
                type="button"
                disabled={!currentResults || validating}
                onClick={validateResults}
              >
                {validating ? 'Validating...' : 'Validate Results'}
              </button>
            </div>

            {validationResult && (
              <div className={`${styles.statusSuccess} ${styles.validationResult}`}>
                <strong>Validation:</strong> Baseline RMSE{' '}
                {validationResult.baseline_rmse.toFixed(3)}px &rarr; Corrected{' '}
                {validationResult.corrected_rmse.toFixed(3)}px ({' '}
                {validationResult.improvement_percent.toFixed(1)}% improvement, {validationResult.num_lines}{' '}
                lines)
              </div>
            )}
          </div>
        </div>

        {/* ======================== Load / Save Calibration ======================== */}
        <div className={`${styles.card} ${styles.cardFullwidth}`}>
          <div className={styles.cardHeader}>Load / Save Calibration</div>
          <div className={styles.cardBody}>
            <div className={styles.loadSaveRow}>
              <select
                className={styles.loadCameraSelect}
                value={selectedCameraId}
                onChange={(e) => setSelectedCameraId(e.target.value)}
              >
                <option value="">-- Select camera --</option>
                {cameraIds.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
              <button
                className={styles.btnSecondary}
                type="button"
                disabled={loadingCalibration}
                onClick={loadCalibrationFromRepo}
              >
                {loadingCalibration ? 'Loading...' : 'Load Calibration'}
              </button>
              <button
                className={styles.btnSuccess}
                type="button"
                disabled={!currentResults || saving}
                onClick={saveCalibration}
              >
                {saving ? 'Saving...' : 'Save Calibration'}
              </button>
            </div>
          </div>
        </div>

        {/* ======================== Results ======================== */}
        {currentResults && (
          <div className={`${styles.card} ${styles.cardFullwidth}`}>
            <div className={styles.cardHeader}>Calibration Results</div>
            <div className={styles.cardBody}>
              <div className={styles.resultsHeader}>
                <div>
                  <strong>Quality: </strong>
                  <span className={QUALITY_CLASSES[currentResults.quality] ?? styles.qualityBadge}>
                    {currentResults.quality}
                  </span>
                </div>
                <div className={styles.rmseDisplay}>
                  <strong>RMSE:</strong> {currentResults.overall_rmse.toFixed(3)} pixels
                </div>
              </div>

              {/* Distortion Coefficients */}
              <h4 className={styles.sectionTitle}>Distortion Coefficients</h4>
              <div className={styles.coefficientsDisplay}>
                <div className={styles.coefRadial}>
                  <div className={styles.coefName}>k1</div>
                  <div className={styles.coefValue}>
                    {currentResults.coefficients.k1.toFixed(6)}
                  </div>
                </div>
                <div className={styles.coefRadial}>
                  <div className={styles.coefName}>k2</div>
                  <div className={styles.coefValue}>
                    {currentResults.coefficients.k2.toFixed(6)}
                  </div>
                </div>
                <div className={styles.coefRadial}>
                  <div className={styles.coefName}>k3</div>
                  <div className={styles.coefValue}>
                    {currentResults.coefficients.k3.toFixed(6)}
                  </div>
                </div>
                <div className={styles.coefTangential}>
                  <div className={styles.coefName}>p1</div>
                  <div className={styles.coefValue}>
                    {currentResults.coefficients.p1.toFixed(6)}
                  </div>
                </div>
                <div className={styles.coefTangential}>
                  <div className={styles.coefName}>p2</div>
                  <div className={styles.coefValue}>
                    {currentResults.coefficients.p2.toFixed(6)}
                  </div>
                </div>
              </div>

              {/* Stats grid */}
              <div className={styles.resultsGrid}>
                <div className={styles.resultItem}>
                  <div className={styles.resultLabel}>Lines Used</div>
                  <div className={styles.resultValue}>{numLines}</div>
                </div>
                <div className={styles.resultItem}>
                  <div className={styles.resultLabel}>Iterations</div>
                  <div className={styles.resultValue}>{currentResults.iterations}</div>
                </div>
                <div className={styles.resultItem}>
                  <div className={styles.resultLabel}>Initial Error</div>
                  <div className={styles.resultValue}>
                    {currentResults.initial_error.toFixed(4)}
                  </div>
                </div>
                <div className={styles.resultItem}>
                  <div className={styles.resultLabel}>Final Error</div>
                  <div className={styles.resultValue}>
                    {currentResults.final_error.toFixed(4)}
                  </div>
                </div>
                <div className={`${styles.resultItem} ${styles.resultHighlight}`}>
                  <div className={styles.resultLabel}>Improvement</div>
                  <div className={styles.resultValue}>
                    {currentResults.improvement_percent.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Optimized intrinsics */}
              {currentResults.optimized_intrinsics && (
                <div className={styles.statusSuccess} style={{ marginTop: 12 }}>
                  <strong>Optimized Intrinsics: </strong>
                  fx={currentResults.optimized_intrinsics.fx.toFixed(1)}, fy=
                  {currentResults.optimized_intrinsics.fy.toFixed(1)}, cx=
                  {currentResults.optimized_intrinsics.cx.toFixed(1)}, cy=
                  {currentResults.optimized_intrinsics.cy.toFixed(1)}{' '}
                  <button
                    className={`${styles.btnSecondary} ${styles.btnSmall}`}
                    style={{ marginLeft: 12 }}
                    type="button"
                    onClick={applyOptimizedIntrinsics}
                  >
                    Apply to fields
                  </button>
                </div>
              )}

              <div className={styles.actions}>
                <button
                  className={styles.btnSecondary}
                  type="button"
                  onClick={copyCoefficients}
                >
                  Copy Coefficients
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ======================== Per-Line Errors ======================== */}
        {currentResults && Array.isArray(currentResults.line_errors) && currentResults.line_errors.length > 0 && (
          <div className={`${styles.card} ${styles.cardFullwidth}`}>
            <div className={styles.cardHeader}>Per-Line Errors (Top 20)</div>
            <div className={styles.cardBody}>
              <div className={styles.linesList}>
                <table className={styles.linesTable}>
                  <thead>
                    <tr>
                      <th>Line</th>
                      <th>RMSE (px)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentResults.line_errors.slice(0, 20).map((err, i) => {
                      // line_errors can be LineError objects or plain numbers
                      if (typeof err === 'number') {
                        return (
                          <tr key={i}>
                            <td>Line {i + 1}</td>
                            <td className={styles.mono}>{err.toFixed(3)}</td>
                          </tr>
                        );
                      }
                      const lineErr = err as unknown as LineError;
                      return (
                        <tr key={lineErr.line_id ?? i}>
                          <td>{lineErr.line_id ?? `Line ${i + 1}`}</td>
                          <td className={styles.mono}>
                            {lineErr.rmse_pixels?.toFixed(3) ?? '---'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
