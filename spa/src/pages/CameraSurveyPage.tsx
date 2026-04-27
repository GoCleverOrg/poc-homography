import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import styles from './CameraSurveyPage.module.css';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TenantInfo {
  id: string;
  name: string;
  description?: string;
}

interface CameraInfo {
  id: string;
  name: string;
  ip: string;
  model?: string;
}

interface Preset {
  name: string;
  axis: string;
  start: number;
  end: number;
  step: number;
}

interface SessionSummary {
  id: string;
  date: string;
  camera_name?: string;
  camera_id: string;
  axis: string;
  step_count: number;
  status: string;
}

interface CaptureEntry {
  filename: string;
  step_index: number;
  ptz: { pan?: number; tilt?: number; zoom?: number };
}

interface SessionDetail {
  session: { id: string };
  captures: CaptureEntry[];
}

interface SurveyProgress {
  session_id: string;
  status: string;
  step_count: number;
  total_steps: number;
  current_ptz?: { pan?: number; tilt?: number; zoom?: number } | null;
  last_capture_path?: string | null;
}

type Axis = 'pan' | 'tilt' | 'zoom';

// ---------------------------------------------------------------------------
// API base path
// ---------------------------------------------------------------------------

const API = '/camera-evaluation/api';

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CameraSurveyPage() {
  const { credentials } = useAuth();

  // ---------- helpers ----------

  const authHeaders = useCallback((): HeadersInit => {
    if (!credentials) return {};
    const encoded = btoa(`${credentials.username}:${credentials.password}`);
    return { Authorization: `Basic ${encoded}` };
  }, [credentials]);

  const apiFetch = useCallback(
    async (url: string, init?: RequestInit) => {
      const res = await fetch(url, {
        ...init,
        headers: { ...authHeaders(), ...init?.headers },
      });
      return res;
    },
    [authHeaders],
  );

  const apiJson = useCallback(
    async <T,>(url: string, init?: RequestInit): Promise<T> => {
      const res = await apiFetch(url, init);
      const json = (await res.json()) as { status: string; data: T; message?: string };
      if (json.status !== 'success') throw new Error(json.message ?? 'Request failed');
      return json.data;
    },
    [apiFetch],
  );

  // ---------- state: tenants / cameras ----------

  const [tenants, setTenants] = useState<TenantInfo[]>([]);
  const [selectedTenantId, setSelectedTenantId] = useState('');
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState('');
  const [selectedCameraData, setSelectedCameraData] = useState<CameraInfo | null>(null);

  // ---------- state: survey config ----------

  const [axis, setAxis] = useState<Axis>('pan');
  const [presets, setPresets] = useState<Preset[]>([]);
  const [startVal, setStartVal] = useState(0);
  const [endVal, setEndVal] = useState(360);
  const [stepVal, setStepVal] = useState(10);
  const [retryTimeout, setRetryTimeout] = useState(60);
  const [tagsText, setTagsText] = useState('');
  const [restorePtz, setRestorePtz] = useState(true);
  const [fixedPan, setFixedPan] = useState('');
  const [fixedTilt, setFixedTilt] = useState('');
  const [fixedZoom, setFixedZoom] = useState('');

  // ---------- state: survey running ----------

  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [surveyStarting, setSurveyStarting] = useState(false);
  const [showProgress, setShowProgress] = useState(false);
  const [progressStatus, setProgressStatus] = useState('Starting...');
  const [progressSteps, setProgressSteps] = useState('Step 0 of 0');
  const [progressPct, setProgressPct] = useState(0);
  const [ptzPan, setPtzPan] = useState('-');
  const [ptzTilt, setPtzTilt] = useState('-');
  const [ptzZoom, setPtzZoom] = useState('-');
  const [lastCaptureUrl, setLastCaptureUrl] = useState<string | null>(null);
  const [abortDisabled, setAbortDisabled] = useState(true);
  const [showCredentialsWarning, setShowCredentialsWarning] = useState(false);

  // ---------- state: sessions ----------

  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionsError, setSessionsError] = useState(false);

  // ---------- state: session detail ----------

  const [detailVisible, setDetailVisible] = useState(false);
  const [detailSessionId, setDetailSessionId] = useState('');
  const [detailCaptures, setDetailCaptures] = useState<CaptureEntry[]>([]);
  const [detailManifest, setDetailManifest] = useState('');
  const [detailImageCount, setDetailImageCount] = useState(0);

  // ---------- state: delete modal ----------

  const [deleteModalVisible, setDeleteModalVisible] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [deleteInProgress, setDeleteInProgress] = useState(false);

  // ---------- refs for polling ----------

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentSessionRef = useRef<string | null>(null);

  // Keep ref in sync
  useEffect(() => {
    currentSessionRef.current = currentSessionId;
  }, [currentSessionId]);

  // ---------- load tenants on mount ----------

  useEffect(() => {
    if (!credentials) return;
    let cancelled = false;

    apiJson<{ tenants: TenantInfo[] }>(`${API}/tenants/`).then((data) => {
      if (!cancelled) setTenants(data.tenants);
    }).catch((e: unknown) => {
      console.error('Failed to load tenants:', e);
    });

    return () => { cancelled = true; };
  }, [credentials, apiJson]);

  // ---------- load presets on mount ----------

  useEffect(() => {
    if (!credentials) return;
    let cancelled = false;

    apiJson<{ presets: Preset[] }>(`${API}/survey/presets/`).then((data) => {
      if (!cancelled) setPresets(data.presets);
    }).catch((e: unknown) => {
      console.error('Failed to load presets:', e);
    });

    return () => { cancelled = true; };
  }, [credentials, apiJson]);

  // ---------- load sessions ----------

  const loadSessions = useCallback(async () => {
    if (!credentials) return;
    setSessionsLoading(true);
    setSessionsError(false);

    try {
      const data = await apiJson<{ sessions: SessionSummary[] }>(`${API}/survey/sessions/`);
      setSessions(data.sessions);
    } catch (e: unknown) {
      console.error('Failed to load sessions:', e);
      setSessionsError(true);
    } finally {
      setSessionsLoading(false);
    }
  }, [credentials, apiJson]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // ---------- load cameras when tenant changes ----------

  useEffect(() => {
    if (!selectedTenantId || !credentials) {
      setCameras([]);
      setSelectedCameraId('');
      setSelectedCameraData(null);
      return;
    }

    let cancelled = false;
    apiJson<{ cameras: CameraInfo[] }>(
      `${API}/cameras/?tenant_id=${encodeURIComponent(selectedTenantId)}`,
    ).then((data) => {
      if (!cancelled) setCameras(data.cameras);
    }).catch((e: unknown) => {
      console.error('Failed to load cameras:', e);
    });

    return () => { cancelled = true; };
  }, [selectedTenantId, credentials, apiJson]);

  // ---------- fetch PTZ position when camera selected ----------

  const fetchPtzPosition = useCallback(async () => {
    if (!selectedTenantId || !selectedCameraId || !credentials) return;

    try {
      const data = await apiJson<{ pan?: number; tilt?: number; zoom?: number }>(
        `${API}/camera/position/?camera_id=${encodeURIComponent(selectedCameraId)}&tenant_id=${encodeURIComponent(selectedTenantId)}`,
      );
      // Note: the /camera/position/ endpoint returns { pan, tilt, zoom } directly in data
      // (not nested under data.ptz)
      setFixedPan(data.pan?.toFixed(1) ?? '');
      setFixedTilt(data.tilt?.toFixed(1) ?? '');
      setFixedZoom(data.zoom?.toFixed(1) ?? '');
    } catch (e: unknown) {
      console.warn('Error fetching PTZ position:', e);
    }
  }, [selectedTenantId, selectedCameraId, credentials, apiJson]);

  // ---------- event handlers ----------

  const handleTenantChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setSelectedTenantId(val);
    setSelectedCameraId('');
    setSelectedCameraData(null);
    setCameras([]);
  }, []);

  const handleCameraChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      setSelectedCameraId(val);
      const cam = cameras.find((c) => c.id === val) ?? null;
      setSelectedCameraData(cam);
      if (cam) fetchPtzPosition();
    },
    [cameras, fetchPtzPosition],
  );

  const handleAxisChange = useCallback((newAxis: Axis) => {
    setAxis(newAxis);
  }, []);

  const applyPreset = useCallback((preset: Preset) => {
    setStartVal(preset.start);
    setEndVal(preset.end);
    setStepVal(preset.step);
  }, []);

  // ---------- filtered presets ----------

  const filteredPresets = presets.filter((p) => p.axis === axis);

  // ---------- survey start ----------

  const startSurvey = useCallback(async () => {
    if (!selectedCameraId || !selectedTenantId) {
      alert('Please select a camera');
      return;
    }
    if (stepVal <= 0) {
      alert('Step must be greater than 0');
      return;
    }

    const tags = tagsText.split(',').map((t) => t.trim()).filter(Boolean);

    try {
      setSurveyStarting(true);

      const data = await apiJson<{ session_id: string }>(`${API}/survey/start/`, {
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

      setCurrentSessionId(data.session_id);
      setAbortDisabled(false);
      setShowProgress(true);
      setProgressStatus('Starting...');
      setProgressSteps('Step 0 of 0');
      setProgressPct(0);
      setPtzPan('-');
      setPtzTilt('-');
      setPtzZoom('-');
      setLastCaptureUrl(null);

      // Start polling
      pollRef.current = setInterval(() => {
        pollStatus(data.session_id);
      }, 1000);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes('credentials')) {
        setShowCredentialsWarning(true);
      }
      alert('Failed to start survey: ' + msg);
    } finally {
      setSurveyStarting(false);
    }
  }, [
    selectedCameraId,
    selectedTenantId,
    axis,
    startVal,
    endVal,
    stepVal,
    restorePtz,
    retryTimeout,
    tagsText,
    fixedPan,
    fixedTilt,
    fixedZoom,
    apiJson,
  ]);

  // ---------- poll status ----------

  const pollStatus = useCallback(
    async (sessionId: string) => {
      try {
        const data = await apiJson<SurveyProgress>(
          `${API}/survey/${encodeURIComponent(sessionId)}/status/`,
        );

        const pct = data.total_steps > 0 ? (data.step_count / data.total_steps) * 100 : 0;
        setProgressSteps(`Step ${data.step_count} of ${data.total_steps}`);
        setProgressPct(pct);

        if (data.current_ptz) {
          setPtzPan(data.current_ptz.pan != null ? data.current_ptz.pan.toFixed(1) + '°' : '-');
          setPtzTilt(data.current_ptz.tilt != null ? data.current_ptz.tilt.toFixed(1) + '°' : '-');
          setPtzZoom(data.current_ptz.zoom != null ? data.current_ptz.zoom.toFixed(1) + 'x' : '-');
        }

        if (data.last_capture_path && currentSessionRef.current) {
          const filename = data.last_capture_path.split('/').pop();
          if (filename) {
            setLastCaptureUrl(
              `${API}/survey/sessions/${encodeURIComponent(sessionId)}/images/${encodeURIComponent(filename)}`,
            );
          }
        }

        if (data.status === 'completed' || data.status === 'aborted' || data.status === 'failed') {
          setProgressStatus(data.status.charAt(0).toUpperCase() + data.status.slice(1));
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          setCurrentSessionId(null);
          setAbortDisabled(true);
          loadSessions();
        } else {
          setProgressStatus('Running...');
        }
      } catch (e: unknown) {
        console.error('Failed to poll status:', e);
      }
    },
    [apiJson, loadSessions],
  );

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  // ---------- abort survey ----------

  const abortSurvey = useCallback(async () => {
    if (!currentSessionId) return;

    try {
      await apiJson<{ session_id: string }>(
        `${API}/survey/${encodeURIComponent(currentSessionId)}/abort/`,
        { method: 'POST' },
      );
      setProgressStatus('Aborting...');
      setAbortDisabled(true);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      alert('Failed to abort survey: ' + msg);
    }
  }, [currentSessionId, apiJson]);

  // ---------- view session ----------

  const viewSession = useCallback(
    async (sessionId: string) => {
      try {
        const data = await apiJson<SessionDetail>(
          `${API}/survey/sessions/${encodeURIComponent(sessionId)}/`,
        );

        setDetailSessionId(data.session.id.substring(0, 8) + '...');
        setDetailManifest(JSON.stringify(data, null, 2));
        setDetailImageCount(data.captures.length);
        setDetailCaptures(
          data.captures.map((c) => ({
            ...c,
            // tag the session id for image URL construction
            _sessionId: sessionId,
          } as CaptureEntry)),
        );
        setDetailVisible(true);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        alert('Failed to load session: ' + msg);
      }
    },
    [apiJson],
  );

  // ---------- download manifest ----------

  const downloadManifest = useCallback(() => {
    if (!detailSessionId) return;
    // detailSessionId is truncated for display; we need the full id.
    // We find the matching session from our sessions list by prefix.
    const prefix = detailSessionId.replace('...', '');
    const match = sessions.find((s) => s.id.startsWith(prefix));
    if (match) {
      window.open(
        `${API}/survey/sessions/${encodeURIComponent(match.id)}/manifest/`,
        '_blank',
      );
    }
  }, [detailSessionId, sessions]);

  // Full session ID for the detail view (for image URLs)
  const detailFullSessionId = (() => {
    const prefix = detailSessionId.replace('...', '');
    return sessions.find((s) => s.id.startsWith(prefix))?.id ?? '';
  })();

  // ---------- delete session ----------

  const showDeleteModal = useCallback((sessionId: string) => {
    setSessionToDelete(sessionId);
    setDeleteModalVisible(true);
  }, []);

  const hideDeleteModal = useCallback(() => {
    setSessionToDelete(null);
    setDeleteModalVisible(false);
  }, []);

  const confirmDelete = useCallback(async () => {
    if (!sessionToDelete) return;

    try {
      setDeleteInProgress(true);
      await apiJson<{ session_id: string }>(
        `${API}/survey/sessions/${encodeURIComponent(sessionToDelete)}/delete/`,
        { method: 'POST' },
      );
      hideDeleteModal();
      loadSessions();

      // Close detail panel if viewing the deleted session
      if (detailSessionId.startsWith(sessionToDelete.substring(0, 8))) {
        setDetailVisible(false);
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      alert('Failed to delete session: ' + msg);
    } finally {
      setDeleteInProgress(false);
    }
  }, [sessionToDelete, apiJson, hideDeleteModal, loadSessions, detailSessionId]);

  // ---------- computed ----------

  const canStartSurvey = !!selectedCameraId && !currentSessionId && !surveyStarting;

  // ---------- render ----------

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <h1>Camera Survey Tool</h1>
        <p>
          Systematically survey camera surroundings by controlling PTZ axes,
          capturing screenshots, and generating manifests
        </p>
      </div>

      {/* Credentials warning */}
      {showCredentialsWarning && (
        <div className={styles.credentialsWarning}>
          <strong>&#9888; Camera Credentials Not Set</strong>
          <p>Set environment variables before running surveys:</p>
          <code>
            export CAMERA_USERNAME=admin
            <br />
            export CAMERA_PASSWORD=your_password
          </code>
        </div>
      )}

      {/* Camera Selection Panel */}
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <h2>Camera Selection</h2>
        </div>
        <div className={styles.formRow}>
          <div>
            <label className={styles.label} htmlFor="survey-tenant-select">
              Tenant
            </label>
            <select
              id="survey-tenant-select"
              className={styles.select}
              value={selectedTenantId}
              onChange={handleTenantChange}
            >
              <option value="">-- Select a tenant --</option>
              {tenants.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className={styles.label} htmlFor="survey-camera-select">
              Camera
            </label>
            <select
              id="survey-camera-select"
              className={styles.select}
              value={selectedCameraId}
              onChange={handleCameraChange}
              disabled={!selectedTenantId}
            >
              <option value="">
                {selectedTenantId ? '-- Select a camera --' : '-- Select tenant first --'}
              </option>
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.ip})
                </option>
              ))}
            </select>
          </div>
        </div>

        {selectedCameraData && (
          <div className={styles.cameraInfo}>
            <div className={styles.cameraInfoRow}>
              <span className={styles.cameraInfoLabel}>IP Address:</span>
              <span className={styles.cameraInfoValue}>{selectedCameraData.ip}</span>
            </div>
            <div className={styles.cameraInfoRow}>
              <span className={styles.cameraInfoLabel}>Model:</span>
              <span className={styles.cameraInfoValue}>
                {selectedCameraData.model ?? 'Unknown'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Survey Configuration Panel */}
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <h2>Survey Configuration</h2>
        </div>

        {/* Axis selection */}
        <label className={styles.label}>Axis</label>
        <div className={styles.radioGroup}>
          {(['pan', 'tilt', 'zoom'] as Axis[]).map((a) => (
            <label key={a} className={styles.radioLabel}>
              <input
                type="radio"
                name="survey-axis"
                value={a}
                checked={axis === a}
                onChange={() => handleAxisChange(a)}
              />
              {a.charAt(0).toUpperCase() + a.slice(1)}
            </label>
          ))}
        </div>

        {/* Fixed axis values */}
        <div className={styles.fixedAxisContainer}>
          {axis !== 'pan' && (
            <div className={styles.formRow}>
              <div>
                <label className={styles.label} htmlFor="survey-fixed-pan">
                  Pan (fixed)
                </label>
                <input
                  id="survey-fixed-pan"
                  type="number"
                  className={styles.inputNumber}
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
              <div>
                <label className={styles.label} htmlFor="survey-fixed-tilt">
                  Tilt (fixed)
                </label>
                <input
                  id="survey-fixed-tilt"
                  type="number"
                  className={styles.inputNumber}
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
              <div>
                <label className={styles.label} htmlFor="survey-fixed-zoom">
                  Zoom (fixed)
                </label>
                <input
                  id="survey-fixed-zoom"
                  type="number"
                  className={styles.inputNumber}
                  step="1"
                  placeholder="Current position"
                  value={fixedZoom}
                  onChange={(e) => setFixedZoom(e.target.value)}
                />
              </div>
            </div>
          )}
        </div>

        {/* Presets */}
        <label className={styles.label}>Presets</label>
        <div className={styles.presetButtons}>
          {filteredPresets.map((p) => (
            <button
              key={p.name}
              type="button"
              className={`${styles.btnSecondary} ${styles.btnSmall}`}
              onClick={() => applyPreset(p)}
            >
              {p.name}
            </button>
          ))}
          {filteredPresets.length === 0 && (
            <span style={{ color: '#666', fontSize: 13 }}>No presets for this axis</span>
          )}
        </div>

        {/* Range inputs */}
        <div className={styles.formRow}>
          <div>
            <label className={styles.label}>Start</label>
            <input
              type="number"
              className={styles.inputNumber}
              value={startVal}
              step="1"
              onChange={(e) => setStartVal(parseFloat(e.target.value) || 0)}
            />
          </div>
          <div>
            <label className={styles.label}>End</label>
            <input
              type="number"
              className={styles.inputNumber}
              value={endVal}
              step="1"
              onChange={(e) => setEndVal(parseFloat(e.target.value) || 0)}
            />
          </div>
          <div>
            <label className={styles.label}>Step</label>
            <input
              type="number"
              className={styles.inputNumber}
              value={stepVal}
              min="0.1"
              step="0.1"
              onChange={(e) => setStepVal(parseFloat(e.target.value) || 0)}
            />
          </div>
        </div>

        {/* Timeout and Tags */}
        <div className={styles.formRow}>
          <div>
            <label className={styles.label}>Retry Timeout (seconds)</label>
            <input
              type="number"
              className={styles.inputNumber}
              value={retryTimeout}
              min="1"
              onChange={(e) => setRetryTimeout(parseInt(e.target.value, 10) || 60)}
            />
          </div>
          <div>
            <label className={styles.label}>Session Tags (comma-separated)</label>
            <input
              type="text"
              className={styles.inputText}
              placeholder="day, sunny, morning"
              value={tagsText}
              onChange={(e) => setTagsText(e.target.value)}
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

      {/* Survey Control Panel */}
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <h2>Survey Control</h2>
        </div>

        <div className={styles.controlButtons}>
          <button
            type="button"
            className={styles.btnPrimary}
            disabled={!canStartSurvey}
            onClick={startSurvey}
          >
            {surveyStarting ? (
              <>
                <span className={styles.spinner} />
                Starting...
              </>
            ) : currentSessionId ? (
              'Survey Running...'
            ) : (
              'Start Survey'
            )}
          </button>
          <button
            type="button"
            className={styles.btnDanger}
            disabled={abortDisabled}
            onClick={abortSurvey}
          >
            Abort Survey
          </button>
        </div>

        {showProgress && (
          <div className={styles.progressContainer}>
            <div className={styles.progressText}>
              <span>{progressStatus}</span>{' '}
              <span>{progressSteps}</span>
            </div>
            <div className={styles.progressBar}>
              <div
                className={styles.progressBarFill}
                style={{ width: `${progressPct}%` }}
              />
            </div>

            <div className={styles.ptzDisplay}>
              <div className={styles.ptzValue}>
                <div className={styles.ptzValueLabel}>Pan</div>
                <div className={styles.ptzValueNumber}>{ptzPan}</div>
              </div>
              <div className={styles.ptzValue}>
                <div className={styles.ptzValueLabel}>Tilt</div>
                <div className={styles.ptzValueNumber}>{ptzTilt}</div>
              </div>
              <div className={styles.ptzValue}>
                <div className={styles.ptzValueLabel}>Zoom</div>
                <div className={styles.ptzValueNumber}>{ptzZoom}</div>
              </div>
            </div>

            <div className={styles.thumbnailContainer}>
              {lastCaptureUrl ? (
                <img src={lastCaptureUrl} alt="Last capture" />
              ) : (
                <div className={styles.thumbnailPlaceholder}>
                  Last capture will appear here
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Past Sessions Panel */}
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <h2>Past Sessions</h2>
          <button
            type="button"
            className={`${styles.btnSecondary} ${styles.btnSmall}`}
            onClick={loadSessions}
          >
            Refresh
          </button>
        </div>

        <table className={styles.sessionsTable}>
          <thead>
            <tr>
              <th>Date</th>
              <th>Camera</th>
              <th>Axis</th>
              <th>Steps</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {sessionsLoading && (
              <tr>
                <td colSpan={6} className={styles.emptyRow}>
                  Loading sessions...
                </td>
              </tr>
            )}
            {sessionsError && (
              <tr>
                <td colSpan={6} className={styles.errorRow}>
                  Error loading sessions
                </td>
              </tr>
            )}
            {!sessionsLoading && !sessionsError && sessions.length === 0 && (
              <tr>
                <td colSpan={6} className={styles.emptyRow}>
                  No sessions yet
                </td>
              </tr>
            )}
            {!sessionsLoading &&
              !sessionsError &&
              sessions.map((s) => (
                <tr key={s.id}>
                  <td>{s.date}</td>
                  <td>{s.camera_name ?? s.camera_id}</td>
                  <td>{s.axis}</td>
                  <td>{s.step_count}</td>
                  <td>
                    <span
                      className={`${styles.statusBadge} ${statusBadgeClass(s.status)}`}
                    >
                      {s.status}
                    </span>
                  </td>
                  <td>
                    <button
                      type="button"
                      className={`${styles.btnSecondary} ${styles.btnSmall}`}
                      onClick={() => viewSession(s.id)}
                    >
                      View
                    </button>{' '}
                    <button
                      type="button"
                      className={`${styles.btnDanger} ${styles.btnSmall}`}
                      onClick={() => showDeleteModal(s.id)}
                      disabled={s.status === 'running'}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      {/* Session Detail Panel */}
      {detailVisible && (
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <h2>Session Details: {detailSessionId}</h2>
            <button
              type="button"
              className={`${styles.btnSecondary} ${styles.btnSmall}`}
              onClick={() => setDetailVisible(false)}
            >
              Close
            </button>
          </div>

          <div style={{ marginBottom: 16 }}>
            <button
              type="button"
              className={`${styles.btnSecondary} ${styles.btnSmall}`}
              onClick={downloadManifest}
            >
              Download Manifest
            </button>
          </div>

          <h3 className={styles.sectionHeading}>Manifest</h3>
          <div className={styles.manifestDisplay}>{detailManifest}</div>

          <h3 className={styles.sectionHeadingImages}>
            Captured Images ({detailImageCount})
          </h3>
          <div className={styles.imageGallery}>
            {detailCaptures.map((c) => {
              const imageUrl = `${API}/survey/sessions/${encodeURIComponent(detailFullSessionId)}/images/${encodeURIComponent(c.filename)}`;
              return (
                <div
                  key={c.filename}
                  className={styles.galleryItem}
                  onClick={() => window.open(imageUrl, '_blank')}
                >
                  <img src={imageUrl} alt={c.filename} loading="lazy" />
                  <div className={styles.galleryItemInfo}>
                    Step {c.step_index} | Pan: {c.ptz.pan?.toFixed(1) ?? '-'}
                    &deg;
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteModalVisible && (
        <div
          className={styles.modalOverlay}
          onClick={(e) => {
            if (e.target === e.currentTarget) hideDeleteModal();
          }}
        >
          <div className={styles.modal}>
            <div className={styles.modalHeader}>
              <div className={styles.modalIcon}>&#9888;</div>
              <h3>Delete Session</h3>
            </div>
            <div className={styles.modalBody}>
              <p>Are you sure you want to delete this survey session?</p>
              <p style={{ marginTop: 12 }}>
                Session ID:{' '}
                <strong>
                  {sessionToDelete ? sessionToDelete.substring(0, 8) + '...' : ''}
                </strong>
              </p>
              <p className={styles.deleteWarning}>
                This action cannot be undone. All captured images and manifest
                data will be permanently deleted.
              </p>
            </div>
            <div className={styles.modalActions}>
              <button
                type="button"
                className={styles.btnSecondary}
                onClick={hideDeleteModal}
              >
                Cancel
              </button>
              <button
                type="button"
                className={styles.btnDanger}
                disabled={deleteInProgress}
                onClick={confirmDelete}
              >
                {deleteInProgress ? (
                  <>
                    <span className={styles.spinner} />
                    Deleting...
                  </>
                ) : (
                  'Delete Session'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusBadgeClass(status: string): string {
  switch (status) {
    case 'completed':
      return styles.statusCompleted;
    case 'running':
      return styles.statusRunning;
    case 'aborted':
      return styles.statusAborted;
    case 'failed':
      return styles.statusFailed;
    default:
      return '';
  }
}
