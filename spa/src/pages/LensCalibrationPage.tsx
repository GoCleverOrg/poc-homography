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
  }, [selectedCameraId, getIntrinsics, showStatus]);

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

            <button
              className={styles.btnPrimary}
              style={{ marginTop: 12 }}
              type="button"
              disabled={!hasAnnotationLines || calibrating}
              onClick={runCalibration}
            >
              {calibrating ? 'Running...' : 'Run Annotated Lines Calibration'}
            </button>

            <p className={styles.helpText}>
              Load line annotation files with N-point traces of lines that should be
              straight.
              <br />
              The solver optimises distortion coefficients to minimise line curvature
              after undistortion.
            </p>
          </div>
        </div>

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
