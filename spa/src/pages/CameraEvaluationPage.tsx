import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTenant } from '../contexts/TenantContext';
import styles from './CameraEvaluationPage.module.css';

// ------------------------------------------------------------------ types

type SurveyAxis = 'pan' | 'tilt' | 'zoom';

interface Camera {
  id: string;
  name: string;
  ip: string;
  model?: string;
}

interface Preset {
  name: string;
  axis: SurveyAxis;
  start: number;
  end: number;
  step: number;
}

interface AxisLimits {
  min: number;
  max: number;
}

interface Capabilities {
  pan: AxisLimits;
  tilt: AxisLimits;
  zoom: AxisLimits;
  min_step: number;
}

interface PTZPosition {
  pan: number | null;
  tilt: number | null;
  zoom: number | null;
}

interface SurveySession {
  id: string;
  date?: string;
  start_time?: string;
  tenant_name?: string;
  tenant_id?: string;
  camera_name?: string;
  camera_id: string;
  axis: string;
  step_count?: number;
  captures?: { length: number }[];
  status: string;
}

interface SessionCapture {
  filename: string;
  step_index: number;
  ptz?: { pan?: number; tilt?: number; zoom?: number };
}

interface SessionDetail {
  captures?: SessionCapture[];
  [key: string]: unknown;
}

// ------------------------------------------------------------------ helpers

const API_BASE = '/camera-evaluation/api';

const STATUS_CLASS: Record<string, string> = {
  completed: styles.statusCompleted,
  running: styles.statusRunning,
  failed: styles.statusFailed,
  aborted: styles.statusAborted,
  pending: styles.statusPending,
};

// ----------------------------------------------------------- component

export default function CameraEvaluationPage() {
  const { credentials } = useAuth();
  const { selectedTenantId } = useTenant();

  // Auth header builder
  const authHeaders = useMemo((): Record<string, string> => {
    if (!credentials) return {};
    const encoded = btoa(`${credentials.username}:${credentials.password}`);
    return { Authorization: `Basic ${encoded}` };
  }, [credentials]);

  // ---- Camera state ----
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [camerasLoading, setCamerasLoading] = useState(false);

  // ---- Capabilities and position ----
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [currentPosition, setCurrentPosition] = useState<PTZPosition | null>(null);

  // ---- Survey config ----
  const [axis, setAxis] = useState<SurveyAxis>('pan');
  const [startVal, setStartVal] = useState(0);
  const [endVal, setEndVal] = useState(360);
  const [stepVal, setStepVal] = useState(10);
  const [retryTimeout, setRetryTimeout] = useState(60);
  const [sessionTags, setSessionTags] = useState('');
  const [restorePtz, setRestorePtz] = useState(true);
  const [fixedPan, setFixedPan] = useState('');
  const [fixedTilt, setFixedTilt] = useState('');
  const [fixedZoom, setFixedZoom] = useState('');
  const [presets, setPresets] = useState<Preset[]>([]);

  // ---- Validation errors ----
  const [startError, setStartError] = useState('');
  const [endError, setEndError] = useState('');
  const [stepError, setStepError] = useState('');

  // ---- Survey running state ----
  const [surveySessionId, setSurveySessionId] = useState<string | null>(null);
  const [surveyRunning, setSurveyRunning] = useState(false);
  const [surveyStarting, setSurveyStarting] = useState(false);
  const [progressStatus, setProgressStatus] = useState('');
  const [progressSteps, setProgressSteps] = useState('Step 0 of 0');
  const [progressPct, setProgressPct] = useState(0);
  const [surveyPtz, setSurveyPtz] = useState<PTZPosition>({ pan: null, tilt: null, zoom: null });
  const [lastCaptureUrl, setLastCaptureUrl] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollStatusRef = useRef<(sessionId: string) => Promise<void>>(async () => {});

  // ---- Sessions ----
  const [sessions, setSessions] = useState<SurveySession[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState('');

  // ---- Session detail ----
  const [detailSessionId, setDetailSessionId] = useState<string | null>(null);
  const [detailData, setDetailData] = useState<SessionDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  // ============================================================ API helpers

  const apiFetch = useCallback(
    async (path: string, options?: RequestInit) => {
      const res = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers: {
          ...authHeaders,
          ...(options?.headers ?? {}),
        },
      });
      return res;
    },
    [authHeaders],
  );

  const apiJson = useCallback(
    async (path: string, options?: RequestInit) => {
      const res = await apiFetch(path, options);
      const data = await res.json();
      if (data.status !== 'success') {
        throw new Error(data.message || 'API error');
      }
      return data.data;
    },
    [apiFetch],
  );

  // ============================================================ Load cameras

  useEffect(() => {
    if (!selectedTenantId || !credentials) {
      setCameras([]);
      setSelectedCameraId('');
      return;
    }

    let cancelled = false;
    setCamerasLoading(true);

    apiJson(`/cameras/?tenant_id=${encodeURIComponent(selectedTenantId)}`)
      .then((data) => {
        if (cancelled) return;
        setCameras(data.cameras);
      })
      .catch(() => {
        if (!cancelled) setCameras([]);
      })
      .finally(() => {
        if (!cancelled) setCamerasLoading(false);
      });

    return () => { cancelled = true; };
  }, [selectedTenantId, credentials, apiJson]);

  // ============================================================ Load presets

  useEffect(() => {
    if (!credentials) return;
    let cancelled = false;

    apiJson('/survey/presets/')
      .then((data) => {
        if (!cancelled) setPresets(data.presets);
      })
      .catch(() => {
        if (!cancelled) setPresets([]);
      });

    return () => { cancelled = true; };
  }, [credentials, apiJson]);

  // ============================================================ Load sessions

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    setSessionsError('');
    try {
      const data = await apiJson('/survey/sessions/');
      setSessions(data.sessions);
    } catch (e) {
      setSessionsError(e instanceof Error ? e.message : 'Failed to load sessions');
      setSessions([]);
    } finally {
      setSessionsLoading(false);
    }
  }, [apiJson]);

  useEffect(() => {
    if (credentials) loadSessions();
  }, [credentials, loadSessions]);

  // ============================================================ Camera selection

  const selectedCamera = useMemo(
    () => cameras.find((c) => c.id === selectedCameraId) ?? null,
    [cameras, selectedCameraId],
  );

  // Fetch capabilities + position when camera changes
  useEffect(() => {
    if (!selectedCameraId || !selectedTenantId || !credentials) {
      setCapabilities(null);
      setCurrentPosition(null);
      return;
    }

    let cancelled = false;
    const qs = `?tenant_id=${encodeURIComponent(selectedTenantId)}&camera_id=${encodeURIComponent(selectedCameraId)}`;

    // Fetch capabilities
    apiJson(`/camera/capabilities/${qs}`)
      .then((data) => {
        if (!cancelled) setCapabilities(data.capabilities);
      })
      .catch(() => {
        if (!cancelled) setCapabilities(null);
      });

    // Fetch position
    apiJson(`/camera/position/${qs}`)
      .then((data) => {
        if (!cancelled) {
          setCurrentPosition(data);
          // Auto-populate fixed axis values from current position
          if (data.pan != null) setFixedPan(data.pan.toFixed(1));
          if (data.tilt != null) setFixedTilt(data.tilt.toFixed(1));
          if (data.zoom != null) setFixedZoom(data.zoom.toFixed(1));
        }
      })
      .catch(() => {
        if (!cancelled) setCurrentPosition(null);
      });

    return () => { cancelled = true; };
  }, [selectedCameraId, selectedTenantId, credentials, apiJson]);

  // ============================================================ Validation

  const validate = useCallback(() => {
    const limits = capabilities?.[axis];
    const minStep = capabilities?.min_step ?? 0.1;
    const unit = axis === 'zoom' ? 'x' : '°';
    let valid = true;

    // Validate start
    if (limits && (startVal < limits.min || startVal > limits.max)) {
      setStartError(`Start must be between ${limits.min}${unit} and ${limits.max}${unit}`);
      valid = false;
    } else {
      setStartError('');
    }

    // Validate end
    if (limits && (endVal < limits.min || endVal > limits.max)) {
      setEndError(`End must be between ${limits.min}${unit} and ${limits.max}${unit}`);
      valid = false;
    } else {
      setEndError('');
    }

    // Validate step
    if (stepVal < minStep) {
      setStepError(`Step must be at least ${minStep}°`);
      valid = false;
    } else if (stepVal <= 0) {
      setStepError('Step must be greater than 0');
      valid = false;
    } else {
      setStepError('');
    }

    return valid;
  }, [axis, startVal, endVal, stepVal, capabilities]);

  // Re-validate when inputs change
  useEffect(() => {
    validate();
  }, [validate]);

  const isFormValid = startError === '' && endError === '' && stepError === '' && selectedCameraId !== '';

  // ============================================================ Helper text

  const axisLimits = capabilities?.[axis];
  const unit = axis === 'zoom' ? 'x' : '°';
  const startHelper = axisLimits ? `Min: ${axisLimits.min}${unit} / Max: ${axisLimits.max}${unit}` : '';
  const endHelper = axisLimits ? `Min: ${axisLimits.min}${unit} / Max: ${axisLimits.max}${unit}` : '';
  const stepHelper = `Minimum step: ${capabilities?.min_step ?? 0.1}°`;

  // ============================================================ Apply preset

  const applyPreset = useCallback((preset: Preset) => {
    setStartVal(preset.start);
    setEndVal(preset.end);
    setStepVal(preset.step);
  }, []);

  const filteredPresets = useMemo(
    () => presets.filter((p) => p.axis === axis),
    [presets, axis],
  );

  // ============================================================ Start survey

  const startSurvey = useCallback(async () => {
    if (!selectedCameraId || !selectedTenantId) return;
    if (!validate()) return;

    setSurveyStarting(true);

    try {
      const tags = sessionTags.split(',').map((t) => t.trim()).filter(Boolean);

      const data = await apiJson('/survey/start/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: selectedTenantId,
          camera_id: selectedCameraId,
          axis,
          start: startVal,
          end: endVal,
          step: stepVal,
          restore_ptz: restorePtz,
          retry_timeout: retryTimeout,
          session_tags: tags,
          fixed_pan: axis !== 'pan' && fixedPan ? parseFloat(fixedPan) : null,
          fixed_tilt: axis !== 'tilt' && fixedTilt ? parseFloat(fixedTilt) : null,
          fixed_zoom: axis !== 'zoom' && fixedZoom ? parseFloat(fixedZoom) : null,
        }),
      });

      const sessionId = data.session_id;
      setSurveySessionId(sessionId);
      setSurveyRunning(true);
      setProgressStatus('Starting...');
      setProgressSteps('Step 0 of 0');
      setProgressPct(0);
      setSurveyPtz({ pan: null, tilt: null, zoom: null });
      setLastCaptureUrl(null);

      // Start polling
      pollRef.current = setInterval(() => pollStatusRef.current(sessionId), 1000);
    } catch (e) {
      alert(`Failed to start survey: ${e instanceof Error ? e.message : e}`);
    } finally {
      setSurveyStarting(false);
    }
  }, [
    selectedCameraId, selectedTenantId, validate, axis, startVal, endVal,
    stepVal, restorePtz, retryTimeout, sessionTags, fixedPan, fixedTilt,
    fixedZoom, apiJson,
  ]);

  // ============================================================ Poll status

  const pollStatus = useCallback(
    async (sessionId: string) => {
      try {
        const data = await apiJson(`/survey/${sessionId}/status/`);
        const pct = data.total_steps > 0
          ? (data.step_count / data.total_steps) * 100
          : 0;

        setProgressSteps(`Step ${data.step_count} of ${data.total_steps}`);
        setProgressPct(pct);

        if (data.current_ptz) {
          setSurveyPtz({
            pan: data.current_ptz.pan ?? null,
            tilt: data.current_ptz.tilt ?? null,
            zoom: data.current_ptz.zoom ?? null,
          });
        }

        if (data.last_capture_path && sessionId) {
          const filename = data.last_capture_path.split('/').pop();
          setLastCaptureUrl(
            `${API_BASE}/survey/sessions/${sessionId}/images/${filename}`,
          );
        }

        if (data.status === 'completed' || data.status === 'aborted' || data.status === 'failed') {
          setProgressStatus(data.status.charAt(0).toUpperCase() + data.status.slice(1));
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          setSurveySessionId(null);
          setSurveyRunning(false);
          loadSessions();
        } else {
          setProgressStatus('Running...');
        }
      } catch {
        // Silently retry on next poll
      }
    },
    [apiJson, loadSessions],
  );

  useEffect(() => {
    pollStatusRef.current = pollStatus;
  }, [pollStatus]);

  // Cleanup poll on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  // ============================================================ Abort survey

  const abortSurvey = useCallback(async () => {
    if (!surveySessionId) return;
    try {
      await apiJson(`/survey/${surveySessionId}/abort/`, { method: 'POST' });
      setProgressStatus('Aborting...');
    } catch (e) {
      alert(`Failed to abort survey: ${e instanceof Error ? e.message : e}`);
    }
  }, [surveySessionId, apiJson]);

  // ============================================================ View session

  const viewSession = useCallback(
    async (sessionId: string) => {
      setDetailLoading(true);
      setDetailSessionId(sessionId);
      try {
        const data = await apiJson(`/survey/sessions/${sessionId}/`);
        setDetailData(data);
      } catch (e) {
        alert(`Failed to load session: ${e instanceof Error ? e.message : e}`);
        setDetailSessionId(null);
      } finally {
        setDetailLoading(false);
      }
    },
    [apiJson],
  );

  // ============================================================ Delete session

  const deleteSession = useCallback(
    async (sessionId: string) => {
      if (!window.confirm('Are you sure you want to delete this survey session? All captured images will be deleted.')) return;
      try {
        await apiJson(`/survey/sessions/${sessionId}/delete/`, { method: 'POST' });
        loadSessions();
        if (detailSessionId === sessionId) {
          setDetailSessionId(null);
          setDetailData(null);
        }
      } catch (e) {
        alert(`Failed to delete session: ${e instanceof Error ? e.message : e}`);
      }
    },
    [apiJson, loadSessions, detailSessionId],
  );

  // ============================================================ Download manifest

  const downloadManifest = useCallback(
    (sessionId: string) => {
      // Build URL with auth - the browser will handle the download.
      // Since this is a protected endpoint, we open it via a constructed link
      // (the auth middleware on the client side will not help with window.open).
      // Instead, fetch the content and trigger a download.
      apiFetch(`/survey/sessions/${sessionId}/manifest/`)
        .then(async (res) => {
          if (!res.ok) throw new Error('Failed to download');
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `survey_${sessionId}_manifest.yaml`;
          a.click();
          URL.revokeObjectURL(url);
        })
        .catch((e) => {
          alert(`Failed to download manifest: ${e instanceof Error ? e.message : e}`);
        });
    },
    [apiFetch],
  );

  // ============================================================ Render

  if (!selectedTenantId) {
    return (
      <div className={styles.page}>
        <Link to="/" className={styles.backLink}>&larr; Back to Tools</Link>
        <div className={styles.header}>
          <h1>Camera Evaluation Tool</h1>
        </div>
        <p style={{ padding: 16 }}>Select a tenant to continue.</p>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <Link to="/" className={styles.backLink}>&larr; Back to Tools</Link>

      <div className={styles.header}>
        <h1>Camera Evaluation Tool</h1>
        <p>Survey camera capabilities with systematic PTZ sweeps and screenshot capture</p>
      </div>

      {/* ============================================ Camera Selection */}
      <div className={styles.panel}>
        <h2>Camera Selection</h2>
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label>Tenant</label>
            <span className={styles.tenantLabel}>{selectedTenantId}</span>
          </div>
          <div className={styles.formGroup}>
            <label>Camera</label>
            <select
              value={selectedCameraId}
              onChange={(e) => setSelectedCameraId(e.target.value)}
              disabled={camerasLoading}
            >
              <option value="">
                {camerasLoading ? '-- Loading cameras --' : '-- Select camera --'}
              </option>
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.ip})
                </option>
              ))}
            </select>
          </div>
        </div>

        {selectedCamera && (
          <div className={styles.cameraInfo}>
            <div className={styles.cameraInfoRow}>
              <span className={styles.cameraInfoLabel}>IP Address:</span>
              <span className={styles.cameraInfoValue}>{selectedCamera.ip}</span>
            </div>
            <div className={styles.cameraInfoRow}>
              <span className={styles.cameraInfoLabel}>Model:</span>
              <span className={styles.cameraInfoValue}>
                {selectedCamera.model || 'Unknown'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* ============================================ Survey Configuration */}
      {selectedCameraId && (
        <div className={styles.panel}>
          <h2>Survey Configuration</h2>

          {/* Axis selection */}
          <div className={styles.formGroup} style={{ marginBottom: 16 }}>
            <label>Axis</label>
            <div className={styles.radioGroup}>
              {(['pan', 'tilt', 'zoom'] as SurveyAxis[]).map((a) => (
                <label key={a} className={styles.radioLabel}>
                  <input
                    type="radio"
                    name="survey-axis"
                    value={a}
                    checked={axis === a}
                    onChange={() => setAxis(a)}
                  />
                  {a.charAt(0).toUpperCase() + a.slice(1)}
                </label>
              ))}
            </div>
          </div>

          {/* Fixed axis values */}
          <div style={{ marginBottom: 12 }}>
            {axis !== 'pan' && (
              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label>Pan (fixed)</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="Current position"
                    value={fixedPan}
                    onChange={(e) => setFixedPan(e.target.value)}
                  />
                </div>
              </div>
            )}
            {axis !== 'tilt' && (
              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label>Tilt (fixed)</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="Current position"
                    value={fixedTilt}
                    onChange={(e) => setFixedTilt(e.target.value)}
                  />
                </div>
              </div>
            )}
            {axis !== 'zoom' && (
              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label>Zoom (fixed)</label>
                  <input
                    type="number"
                    step="1"
                    placeholder="Current position"
                    value={fixedZoom}
                    onChange={(e) => setFixedZoom(e.target.value)}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Current position display */}
          {currentPosition && (
            <div className={styles.currentPosition}>
              <div className={styles.currentPositionTitle}>Current Position</div>
              <div className={styles.cameraInfoRow}>
                <span className={styles.cameraInfoLabel}>Pan:</span>
                <span className={styles.cameraInfoValue}>
                  {currentPosition.pan != null ? `${currentPosition.pan.toFixed(1)}°` : '-'}
                </span>
              </div>
              <div className={styles.cameraInfoRow}>
                <span className={styles.cameraInfoLabel}>Tilt:</span>
                <span className={styles.cameraInfoValue}>
                  {currentPosition.tilt != null ? `${currentPosition.tilt.toFixed(1)}°` : '-'}
                </span>
              </div>
              <div className={styles.cameraInfoRow}>
                <span className={styles.cameraInfoLabel}>Zoom:</span>
                <span className={styles.cameraInfoValue}>
                  {currentPosition.zoom != null ? `${currentPosition.zoom.toFixed(1)}x` : '-'}
                </span>
              </div>
            </div>
          )}

          {/* Presets */}
          {filteredPresets.length > 0 && (
            <div className={styles.formGroup} style={{ marginBottom: 16 }}>
              <label>Presets</label>
              <div className={styles.presetButtons}>
                {filteredPresets.map((p) => (
                  <button
                    key={p.name}
                    type="button"
                    className={styles.presetBtn}
                    onClick={() => applyPreset(p)}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Start / End / Step */}
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label>Start</label>
              <input
                type="number"
                step="1"
                value={startVal}
                onChange={(e) => setStartVal(parseFloat(e.target.value) || 0)}
                className={startError ? styles.inputError : ''}
              />
              {startHelper && <small className={styles.helperText}>{startHelper}</small>}
              {startError && <small className={styles.errorMessage}>{startError}</small>}
            </div>
            <div className={styles.formGroup}>
              <label>End</label>
              <input
                type="number"
                step="1"
                value={endVal}
                onChange={(e) => setEndVal(parseFloat(e.target.value) || 0)}
                className={endError ? styles.inputError : ''}
              />
              {endHelper && <small className={styles.helperText}>{endHelper}</small>}
              {endError && <small className={styles.errorMessage}>{endError}</small>}
            </div>
            <div className={styles.formGroup}>
              <label>Step</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={stepVal}
                onChange={(e) => setStepVal(parseFloat(e.target.value) || 0)}
                className={stepError ? styles.inputError : ''}
              />
              <small className={styles.helperText}>{stepHelper}</small>
              {stepError && <small className={styles.errorMessage}>{stepError}</small>}
            </div>
          </div>

          {/* Timeout / Tags */}
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label>Retry Timeout (seconds)</label>
              <input
                type="number"
                min="1"
                value={retryTimeout}
                onChange={(e) => setRetryTimeout(parseInt(e.target.value) || 60)}
              />
            </div>
            <div className={styles.formGroup}>
              <label>Session Tags (comma-separated)</label>
              <input
                type="text"
                placeholder="day, sunny, morning"
                value={sessionTags}
                onChange={(e) => setSessionTags(e.target.value)}
              />
            </div>
          </div>

          {/* Restore PTZ */}
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={restorePtz}
              onChange={(e) => setRestorePtz(e.target.checked)}
            />
            Restore PTZ position after survey
          </label>
        </div>
      )}

      {/* ============================================ Survey Control */}
      {selectedCameraId && (
        <div className={styles.panel}>
          <h2>Survey Control</h2>

          <div className={styles.controlButtons}>
            <button
              type="button"
              className={`${styles.btn} ${styles.btnPrimary}`}
              disabled={!isFormValid || surveyRunning || surveyStarting}
              onClick={startSurvey}
            >
              {surveyStarting ? (
                <><span className={styles.spinner} /> Starting...</>
              ) : surveyRunning ? (
                'Survey Running...'
              ) : (
                'Start Survey'
              )}
            </button>
            <button
              type="button"
              className={`${styles.btn} ${styles.btnDanger}`}
              disabled={!surveyRunning}
              onClick={abortSurvey}
            >
              Abort Survey
            </button>
          </div>

          {/* Progress display */}
          {(surveyRunning || progressStatus) && (
            <div className={styles.progressContainer}>
              <div className={styles.progressText}>
                <span>{progressStatus}</span>
                <span>{progressSteps}</span>
              </div>
              <div className={styles.progressBar}>
                <div
                  className={styles.progressBarFill}
                  style={{ width: `${progressPct}%` }}
                />
              </div>

              {/* PTZ during survey */}
              <div className={styles.ptzDisplay}>
                <div className={styles.ptzAxis}>
                  <div className={styles.ptzAxisLabel}>Pan</div>
                  <div className={styles.ptzAxisValue}>
                    {surveyPtz.pan != null ? `${surveyPtz.pan.toFixed(1)}°` : '-'}
                  </div>
                </div>
                <div className={styles.ptzAxis}>
                  <div className={styles.ptzAxisLabel}>Tilt</div>
                  <div className={styles.ptzAxisValue}>
                    {surveyPtz.tilt != null ? `${surveyPtz.tilt.toFixed(1)}°` : '-'}
                  </div>
                </div>
                <div className={styles.ptzAxis}>
                  <div className={styles.ptzAxisLabel}>Zoom</div>
                  <div className={styles.ptzAxisValue}>
                    {surveyPtz.zoom != null ? `${surveyPtz.zoom.toFixed(1)}x` : '-'}
                  </div>
                </div>
              </div>

              {/* Last capture thumbnail */}
              <div className={styles.thumbnailContainer}>
                {lastCaptureUrl ? (
                  <img
                    src={lastCaptureUrl}
                    alt="Last capture"
                    className={styles.thumbnailImg}
                  />
                ) : (
                  <div className={styles.thumbnailPlaceholder}>
                    Last capture will appear here
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============================================ Past Sessions */}
      <div className={styles.panel}>
        <h2>Past Survey Sessions</h2>

        {sessionsLoading && (
          <div className={styles.emptyState}>
            <span className={styles.spinner} /> Loading sessions...
          </div>
        )}

        {!sessionsLoading && sessionsError && (
          <div className={styles.emptyState}>
            Failed to load sessions: {sessionsError}
          </div>
        )}

        {!sessionsLoading && !sessionsError && sessions.length === 0 && (
          <div className={styles.emptyState}>
            No survey sessions found. Run a survey to see results here.
          </div>
        )}

        {!sessionsLoading && sessions.length > 0 && (
          <table className={styles.sessionsTable}>
            <thead>
              <tr>
                <th>Date</th>
                <th>Tenant</th>
                <th>Camera</th>
                <th>Axis</th>
                <th>Steps</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id}>
                  <td>
                    {s.date || (s.start_time ? new Date(s.start_time).toLocaleString() : '--')}
                  </td>
                  <td>{s.tenant_name || s.tenant_id || '--'}</td>
                  <td>{s.camera_name || s.camera_id}</td>
                  <td>{s.axis}</td>
                  <td>{s.step_count ?? (Array.isArray(s.captures) ? s.captures.length : 0)}</td>
                  <td>
                    <span className={`${styles.statusBadge} ${STATUS_CLASS[s.status] ?? styles.statusPending}`}>
                      {s.status}
                    </span>
                  </td>
                  <td className={styles.actions}>
                    <button
                      type="button"
                      className={`${styles.btn} ${styles.btnSm} ${styles.btnPrimary}`}
                      onClick={() => viewSession(s.id)}
                    >
                      View
                    </button>
                    <button
                      type="button"
                      className={`${styles.btn} ${styles.btnSm} ${styles.btnDanger}`}
                      disabled={s.status === 'running'}
                      onClick={() => deleteSession(s.id)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* ============================================ Session Detail */}
      {detailSessionId && (
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <h2>
              Session Details: {detailSessionId.substring(0, 8)}...
            </h2>
            <button
              type="button"
              className={`${styles.btn} ${styles.btnSm} ${styles.btnSecondary}`}
              onClick={() => {
                setDetailSessionId(null);
                setDetailData(null);
              }}
            >
              Close
            </button>
          </div>

          {detailLoading && (
            <div className={styles.emptyState}>
              <span className={styles.spinner} /> Loading session details...
            </div>
          )}

          {!detailLoading && detailData && (
            <>
              <div style={{ marginBottom: 16 }}>
                <button
                  type="button"
                  className={`${styles.btn} ${styles.btnSm} ${styles.btnSecondary}`}
                  onClick={() => downloadManifest(detailSessionId)}
                >
                  Download Manifest
                </button>
              </div>

              <h3 style={{ marginBottom: 12, fontSize: 14, color: '#888' }}>Manifest</h3>
              <div className={styles.manifestDisplay}>
                {JSON.stringify(detailData, null, 2)}
              </div>

              <h3 style={{ margin: '20px 0 12px', fontSize: 14, color: '#888' }}>
                Captured Images ({detailData.captures?.length ?? 0})
              </h3>
              <div className={styles.imageGallery}>
                {(detailData.captures ?? []).map((c) => {
                  const imgUrl = `${API_BASE}/survey/sessions/${detailSessionId}/images/${c.filename}`;
                  return (
                    <div
                      key={c.filename}
                      className={styles.galleryItem}
                      onClick={() => window.open(imgUrl, '_blank')}
                    >
                      <img src={imgUrl} alt={c.filename} loading="lazy" />
                      <div className={styles.galleryItemInfo}>
                        Step {c.step_index} | Pan: {c.ptz?.pan?.toFixed(1) ?? '-'}&deg;
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
