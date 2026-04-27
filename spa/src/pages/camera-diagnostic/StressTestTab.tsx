/**
 * Stress Testing tab -- preset configuration, test execution with status
 * polling, evaluation, and session history.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  Camera,
  AxisConfig,
  StressPreset,
  StressSessionSummary,
} from './api';
import {
  fetchCameras,
  fetchStressPresets,
  startStressTest as apiStartStressTest,
  fetchStressStatus,
  abortStressTest as apiAbortStressTest,
  fetchStressSessions,
  fetchStressSessionDetail,
  submitStressEvaluation,
  deleteStressSession,
  stressVideoStreamUrl,
} from './api';
import styles from '../CameraDiagnosticPage.module.css';

/* ------------------------------------------------------------------ */
/* Types                                                              */
/* ------------------------------------------------------------------ */

interface AxisFormState {
  enabled: boolean;
  start: number;
  end: number;
  step: number;
  useRandomSteps: boolean;
  stepMin: number;
  stepMax: number;
}

const DEFAULT_PAN: AxisFormState = {
  enabled: true,
  start: 0,
  end: 10,
  step: 10,
  useRandomSteps: false,
  stepMin: 5,
  stepMax: 15,
};
const DEFAULT_TILT: AxisFormState = {
  enabled: false,
  start: 0,
  end: 5,
  step: 5,
  useRandomSteps: false,
  stepMin: 5,
  stepMax: 15,
};

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

function formatEvaluation(value: string): string {
  const map: Record<string, string> = {
    good: 'Good',
    needs_improvement: 'Needs Work',
    bad: 'Bad',
    not_evaluated: 'Not Rated',
  };
  return map[value] ?? value;
}

function axisFormToConfig(
  axis: string,
  form: AxisFormState,
): Record<string, unknown> {
  return {
    axis,
    start: form.start,
    end: form.end,
    step: form.step,
    use_random_steps: form.useRandomSteps,
    step_min: form.stepMin,
    step_max: form.stepMax,
  };
}

function axisConfigToForm(
  config: AxisConfig | null,
  defaults: AxisFormState,
): AxisFormState {
  if (!config) return { ...defaults, enabled: false };
  return {
    enabled: true,
    start: config.start,
    end: config.end,
    step: config.step,
    useRandomSteps: config.use_random_steps,
    stepMin: config.step_min,
    stepMax: config.step_max,
  };
}

/* ------------------------------------------------------------------ */
/* Props                                                              */
/* ------------------------------------------------------------------ */

interface Props {
  tenantId: string;
  tenantName: string;
}

/* ------------------------------------------------------------------ */
/* Component                                                          */
/* ------------------------------------------------------------------ */

export default function StressTestTab({ tenantId, tenantName }: Props) {
  /* ---- cameras ---- */
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  /* ---- presets ---- */
  const [presets, setPresets] = useState<StressPreset[]>([]);
  const [selectedPresetIdx, setSelectedPresetIdx] = useState<number | null>(
    null,
  );

  /* ---- form state ---- */
  const [testType, setTestType] = useState('oscillation');
  const [repetitions, setRepetitions] = useState(10);
  const [pan, setPan] = useState<AxisFormState>(DEFAULT_PAN);
  const [tilt, setTilt] = useState<AxisFormState>(DEFAULT_TILT);

  /* ---- running state ---- */
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const [progressText, setProgressText] = useState('');
  const [ptzPosition, setPtzPosition] = useState<{
    pan?: number;
    tilt?: number;
    zoom?: number;
  } | null>(null);

  /* ---- results ---- */
  const [showResults, setShowResults] = useState(false);
  const [resultStatus, setResultStatus] = useState('--');
  const [resultPositionMatch, setResultPositionMatch] = useState('--');
  const [resultPositionError, setResultPositionError] = useState('--');
  const [resultDuration, setResultDuration] = useState('--');
  const [resultMovements, setResultMovements] = useState('--');

  /* ---- evaluation ---- */
  const [currentEvaluation, setCurrentEvaluation] = useState<string | null>(
    null,
  );
  const [evalNotes, setEvalNotes] = useState('');
  const [evalEnabled, setEvalEnabled] = useState(false);

  /* ---- batch run state ---- */
  const batchQueueRef = useRef<number[]>([]);
  const batchTotalRef = useRef(0);
  const batchResultsRef = useRef<
    { preset: string; status: string; sessionId?: string }[]
  >([]);
  const isBatchRef = useRef(false);
  const isMaxSpeedBatchRef = useRef(false);

  /* ---- multi-camera batch state ---- */
  const multiCamQueueRef = useRef<string[]>([]);
  const multiCamTotalRef = useRef(0);
  const multiCamResultsRef = useRef<
    {
      camera: { id: string; name: string };
      presets: { preset: string; status: string; error?: string }[];
    }[]
  >([]);
  const isMultiCamRef = useRef(false);
  const isMultiCamMaxSpeedRef = useRef(false);

  /* ---- button labels ---- */
  const [runAllLabel, setRunAllLabel] = useState('Run All Tests');
  const [runMaxLabel, setRunMaxLabel] = useState('Run Max Speed Tests');
  const [runAllCamAllLabel, setRunAllCamAllLabel] = useState(
    'Run All Cameras (All Presets)',
  );
  const [runAllCamMaxLabel, setRunAllCamMaxLabel] = useState(
    'Run All Cameras (Max Speed)',
  );
  const [allButtonsDisabled, setAllButtonsDisabled] = useState(false);

  /* ---- sessions list ---- */
  const [sessions, setSessions] = useState<StressSessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* ---- load cameras ---- */
  useEffect(() => {
    let cancelled = false;
    fetchCameras(tenantId).then((res) => {
      if (cancelled) return;
      if (res?.status === 'success' && res.data) {
        setCameras(res.data.cameras);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [tenantId]);

  /* ---- load presets on mount ---- */
  const loadPresets = useCallback(async () => {
    try {
      const res = await fetchStressPresets();
      if (res?.status === 'success' && res.data) {
        setPresets(res.data.presets);
        return res.data.presets;
      }
    } catch {
      /* ignore */
    }
    return [] as StressPreset[];
  }, []);

  useEffect(() => {
    loadPresets();
  }, [loadPresets]);

  /* ---- load sessions ---- */
  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const res = await fetchStressSessions(
        tenantId || undefined,
        selectedCamera || undefined,
      );
      if (res?.status === 'success' && res.data) {
        setSessions(res.data.sessions);
      }
    } catch {
      /* ignore */
    } finally {
      setSessionsLoading(false);
    }
  }, [tenantId, selectedCamera]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  /* ---- cleanup poll on unmount ---- */
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  /* ---- select preset ---- */
  const selectPreset = useCallback(
    (index: number) => {
      setSelectedPresetIdx(index);
      const preset = presets[index];
      if (!preset) return;
      setTestType(preset.test_type);
      setRepetitions(preset.repetitions);
      setPan(axisConfigToForm(preset.pan_config, DEFAULT_PAN));
      setTilt(axisConfigToForm(preset.tilt_config, DEFAULT_TILT));
    },
    [presets],
  );

  /* ---- build config from form ---- */
  const buildConfig = useCallback(() => {
    const config: Record<string, unknown> = {
      tenant_id: tenantId,
      camera_id: selectedCamera,
      test_type: testType,
      repetitions,
    };
    if (pan.enabled) config.pan_config = axisFormToConfig('pan', pan);
    if (tilt.enabled) config.tilt_config = axisFormToConfig('tilt', tilt);
    return config;
  }, [tenantId, selectedCamera, testType, repetitions, pan, tilt]);

  /* ---- show results from session detail ---- */
  const showSessionResults = useCallback(
    (session: {
      status: string;
      result?: {
        success: boolean;
        position_match: boolean;
        position_error: { pan?: number; tilt?: number; zoom?: number };
        total_duration_ms: number;
        movements?: unknown[];
      } | null;
      user_evaluation?: string;
      user_notes?: string;
    }) => {
      setShowResults(true);
      setResultStatus(session.status);
      if (session.result) {
        const r = session.result;
        setResultPositionMatch(r.position_match ? 'Yes' : 'No');
        const err = r.position_error;
        setResultPositionError(
          `${err.pan?.toFixed(2) ?? '--'}deg / ${err.tilt?.toFixed(2) ?? '--'}deg / ${err.zoom?.toFixed(2) ?? '--'}`,
        );
        setResultDuration((r.total_duration_ms / 1000).toFixed(1) + 's');
        setResultMovements(String(r.movements?.length ?? 0));
      }
      setEvalEnabled(true);
      if (
        session.user_evaluation &&
        session.user_evaluation !== 'not_evaluated'
      ) {
        setCurrentEvaluation(session.user_evaluation);
      }
      setEvalNotes(session.user_notes ?? '');
    },
    [],
  );

  /* ---- on test complete ---- */
  const onTestComplete = useCallback(async () => {
    if (!isBatchRef.current && !isMultiCamRef.current) {
      setIsRunning(false);
    }

    // Load session detail
    let sessionResult: {
      status: string;
      result?: {
        success: boolean;
        position_match: boolean;
        position_error: { pan?: number; tilt?: number; zoom?: number };
        total_duration_ms: number;
        movements?: unknown[];
      } | null;
      user_evaluation?: string;
      user_notes?: string;
    } | null = null;

    if (currentSessionId) {
      try {
        const res = await fetchStressSessionDetail(currentSessionId);
        if (res?.status === 'success' && res.data) {
          sessionResult = res.data.session;
          if (!isBatchRef.current) showSessionResults(sessionResult);
        }
      } catch {
        /* ignore */
      }
    }

    loadSessions();

    // Batch mode continuation
    if (isBatchRef.current) {
      if (sessionResult && currentSessionId) {
        batchResultsRef.current.push({
          preset:
            presets[selectedPresetIdx ?? 0]?.name ?? 'Unknown',
          status: sessionResult.status,
          sessionId: currentSessionId,
        });
      }

      if (batchQueueRef.current.length > 0) {
        setTimeout(() => runNextInBatchQueue(), 1000);
      } else if (isMultiCamRef.current) {
        // Collect results for this camera
        const camera = cameras.find((c) => c.id === selectedCamera);
        multiCamResultsRef.current.push({
          camera: {
            id: selectedCamera ?? '',
            name: camera?.name ?? selectedCamera ?? '',
          },
          presets: batchResultsRef.current.map((r) => ({
            preset: r.preset,
            status: r.status,
            error: r.status !== 'completed' ? 'Test failed' : undefined,
          })),
        });
        isBatchRef.current = false;
        isMaxSpeedBatchRef.current = false;
        batchResultsRef.current = [];
        setTimeout(() => runNextCameraInQueue(), 1000);
      } else {
        // Single-camera batch completed
        const passedCount = batchResultsRef.current.filter(
          (r) => r.status === 'completed',
        ).length;
        const failedCount = batchResultsRef.current.filter(
          (r) => r.status === 'failed',
        ).length;
        alert(
          `${batchResultsRef.current.length} tests completed!\n\nPassed: ${passedCount}\nFailed: ${failedCount}`,
        );
        resetAllButtons();
      }
    }
  }, [
    currentSessionId,
    loadSessions,
    presets,
    selectedPresetIdx,
    showSessionResults,
    cameras,
    selectedCamera,
  ]);

  // Keep onTestComplete in a ref so the poll callback uses the latest version
  const onTestCompleteRef = useRef(onTestComplete);
  onTestCompleteRef.current = onTestComplete;

  /* ---- poll status ---- */
  const startPolling = useCallback((sessionId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetchStressStatus(sessionId);
        if (res?.status !== 'success' || !res.data) return;
        const prog = res.data;

        const pct =
          (prog.total_movements ?? 0) > 0
            ? Math.round(
                ((prog.current_movement ?? 0) / (prog.total_movements ?? 1)) *
                  100,
              )
            : 0;
        setProgressPct(pct);
        setProgressText(
          `${prog.message} (Rep ${prog.current_repetition}/${prog.total_repetitions}, Move ${prog.current_movement ?? 0}/${prog.total_movements ?? 0})`,
        );
        if (prog.current_position) setPtzPosition(prog.current_position);

        if (
          prog.completed ||
          ['completed', 'failed', 'aborted'].includes(prog.status)
        ) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          onTestCompleteRef.current();
        }
      } catch {
        /* ignore */
      }
    }, 500);
  }, []);

  /* ---- start test ---- */
  const doStartTest = useCallback(async () => {
    const config = buildConfig();
    setIsRunning(true);
    setShowResults(false);
    setProgressPct(0);
    setProgressText('Initializing...');
    setPtzPosition(null);

    try {
      const res = await apiStartStressTest(
        config as Parameters<typeof apiStartStressTest>[0],
      );
      if (res?.status !== 'success' || !res.data) {
        throw new Error(
          (res as unknown as Record<string, unknown>)?.message as string ??
            'Failed to start test',
        );
      }
      const sessionId = res.data.session_id;
      setCurrentSessionId(sessionId);
      startPolling(sessionId);
    } catch (e: unknown) {
      alert(`Failed to start test: ${(e as Error).message}`);
      if (isBatchRef.current) {
        if (isMultiCamRef.current) {
          const camera = cameras.find((c) => c.id === selectedCamera);
          multiCamResultsRef.current.push({
            camera: {
              id: selectedCamera ?? '',
              name: camera?.name ?? selectedCamera ?? '',
            },
            presets: [
              {
                preset: 'N/A',
                status: 'failed',
                error: (e as Error).message,
              },
            ],
          });
          isBatchRef.current = false;
          isMaxSpeedBatchRef.current = false;
          batchResultsRef.current = [];
          setTimeout(() => runNextCameraInQueue(), 1000);
        } else {
          resetAllButtons();
        }
      } else {
        setIsRunning(false);
      }
    }
  }, [buildConfig, startPolling, cameras, selectedCamera]);

  // Keep doStartTest ref for batch calls
  const doStartTestRef = useRef(doStartTest);
  doStartTestRef.current = doStartTest;

  /* ---- abort ---- */
  const handleAbort = useCallback(async () => {
    if (currentSessionId) {
      try {
        await apiAbortStressTest(currentSessionId);
      } catch {
        /* ignore */
      }
    }
    if (isMultiCamRef.current) {
      abortMultiCamera();
    } else if (isBatchRef.current) {
      resetAllButtons();
    }
  }, [currentSessionId]);

  /* ---- batch queue runner ---- */
  const runNextInBatchQueue = useCallback(() => {
    if (batchQueueRef.current.length === 0) return;
    const idx = batchQueueRef.current.shift()!;
    selectPreset(idx);
    // Need a tick for state to settle
    setTimeout(() => doStartTestRef.current(), 50);
  }, [selectPreset]);

  /* ---- start run all tests ---- */
  const startRunAllTests = useCallback(() => {
    if (presets.length === 0) return;
    isBatchRef.current = true;
    isMaxSpeedBatchRef.current = false;
    batchQueueRef.current = presets.map((_, i) => i);
    batchResultsRef.current = [];
    batchTotalRef.current = presets.length;
    setAllButtonsDisabled(true);
    setRunAllLabel(`Running All (0/${presets.length})...`);
    runNextInBatchQueue();
  }, [presets, runNextInBatchQueue]);

  /* ---- start run max speed tests ---- */
  const startRunMaxSpeedTests = useCallback(() => {
    const indices = presets
      .map((p, i) => (p.max_speed ? i : -1))
      .filter((i) => i !== -1);
    if (indices.length === 0) return;
    isBatchRef.current = true;
    isMaxSpeedBatchRef.current = true;
    batchQueueRef.current = indices;
    batchResultsRef.current = [];
    batchTotalRef.current = indices.length;
    setAllButtonsDisabled(true);
    setRunMaxLabel(`Running Max Speed (0/${indices.length})...`);
    runNextInBatchQueue();
  }, [presets, runNextInBatchQueue]);

  /* ---- multi-camera ---- */
  const runNextCameraInQueue = useCallback(() => {
    if (multiCamQueueRef.current.length === 0) {
      // All cameras done
      isMultiCamRef.current = false;
      const totalPresets = multiCamResultsRef.current.reduce(
        (s, r) => s + r.presets.length,
        0,
      );
      const totalPassed = multiCamResultsRef.current.reduce(
        (s, r) =>
          s + r.presets.filter((p) => p.status === 'completed').length,
        0,
      );
      const totalFailed = totalPresets - totalPassed;
      resetAllButtons();
      alert(
        `Multi-Camera Test Complete:\n${multiCamTotalRef.current} cameras tested\n${totalPresets} total presets\n${totalPassed} passed, ${totalFailed} failed`,
      );
      multiCamResultsRef.current = [];
      return;
    }

    const cameraId = multiCamQueueRef.current.shift()!;
    setSelectedCamera(cameraId);

    // Build preset queue for this camera
    let presetQueue: number[];
    if (isMultiCamMaxSpeedRef.current) {
      presetQueue = presets
        .map((p, i) => (p.max_speed ? i : -1))
        .filter((i) => i !== -1);
    } else {
      presetQueue = presets.map((_, i) => i);
    }

    if (presetQueue.length === 0) {
      multiCamResultsRef.current.push({
        camera: {
          id: cameraId,
          name: cameras.find((c) => c.id === cameraId)?.name ?? cameraId,
        },
        presets: [],
      });
      setTimeout(() => runNextCameraInQueue(), 500);
      return;
    }

    isBatchRef.current = true;
    isMaxSpeedBatchRef.current = isMultiCamMaxSpeedRef.current;
    batchQueueRef.current = presetQueue;
    batchResultsRef.current = [];
    batchTotalRef.current = presetQueue.length;

    const camIdx =
      multiCamTotalRef.current - multiCamQueueRef.current.length;
    if (isMultiCamMaxSpeedRef.current) {
      setRunAllCamMaxLabel(
        `Running Camera ${camIdx}/${multiCamTotalRef.current}...`,
      );
    } else {
      setRunAllCamAllLabel(
        `Running Camera ${camIdx}/${multiCamTotalRef.current}...`,
      );
    }

    runNextInBatchQueue();
  }, [presets, cameras, runNextInBatchQueue]);

  // Keep ref for async callbacks
  const runNextCameraInQueueRef = useRef(runNextCameraInQueue);
  runNextCameraInQueueRef.current = runNextCameraInQueue;

  const startRunAllCameras = useCallback(
    (maxSpeedOnly: boolean) => {
      if (cameras.length === 0) return;
      isMultiCamRef.current = true;
      isMultiCamMaxSpeedRef.current = maxSpeedOnly;
      multiCamQueueRef.current = cameras.map((c) => c.id);
      multiCamTotalRef.current = cameras.length;
      multiCamResultsRef.current = [];
      setAllButtonsDisabled(true);
      const label = maxSpeedOnly
        ? `Running Camera 1/${cameras.length}...`
        : `Running Camera 1/${cameras.length}...`;
      if (maxSpeedOnly) setRunAllCamMaxLabel(label);
      else setRunAllCamAllLabel(label);
      runNextCameraInQueue();
    },
    [cameras, runNextCameraInQueue],
  );

  const abortMultiCamera = useCallback(() => {
    isMultiCamRef.current = false;
    multiCamQueueRef.current = [];
    isBatchRef.current = false;
    batchQueueRef.current = [];
    resetAllButtons();
  }, []);

  /* ---- reset ---- */
  function resetAllButtons() {
    setAllButtonsDisabled(false);
    setIsRunning(false);
    isBatchRef.current = false;
    isMaxSpeedBatchRef.current = false;
    batchQueueRef.current = [];
    batchResultsRef.current = [];
    setRunAllLabel('Run All Tests');
    setRunMaxLabel('Run Max Speed Tests');
    setRunAllCamAllLabel('Run All Cameras (All Presets)');
    setRunAllCamMaxLabel('Run All Cameras (Max Speed)');
  }

  /* ---- save evaluation ---- */
  const saveEvaluation = useCallback(async () => {
    if (!currentSessionId || !currentEvaluation) return;
    try {
      const res = await submitStressEvaluation(
        currentSessionId,
        currentEvaluation,
        evalNotes,
      );
      if (res?.status === 'success') {
        alert('Evaluation saved!');
        loadSessions();
      }
    } catch {
      /* ignore */
    }
  }, [currentSessionId, currentEvaluation, evalNotes, loadSessions]);

  /* ---- view session ---- */
  const viewSession = useCallback(
    async (sessionId: string) => {
      setCurrentSessionId(sessionId);
      try {
        const res = await fetchStressSessionDetail(sessionId);
        if (res?.status === 'success' && res.data) {
          showSessionResults(res.data.session);
        }
      } catch {
        /* ignore */
      }
    },
    [showSessionResults],
  );

  /* ---- delete session ---- */
  const handleDeleteSession = useCallback(
    async (sessionId: string) => {
      if (!confirm('Are you sure you want to delete this session?')) return;
      try {
        const res = await deleteStressSession(sessionId);
        if (res?.status === 'success') loadSessions();
      } catch {
        /* ignore */
      }
    },
    [loadSessions],
  );

  /* ---- camera change ---- */
  const onCameraChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setSelectedCamera(e.target.value || null);
    },
    [],
  );

  const showPanels = !!selectedCamera;

  /* ---- render ---- */
  return (
    <div>
      {/* Camera selection */}
      <div className={styles.stressPanel}>
        <h2>Camera Selection</h2>
        <div className={styles.stressFormRow}>
          <div className={styles.stressFormGroup}>
            <label>Tenant</label>
            <span className={styles.selectorLabel}>{tenantName}</span>
          </div>
          <div className={styles.stressFormGroup}>
            <label>Camera</label>
            <select
              className={styles.select}
              value={selectedCamera ?? ''}
              onChange={onCameraChange}
              disabled={cameras.length === 0}
            >
              <option value="">-- Select camera --</option>
              {cameras.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.ip})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Multi-camera controls */}
      {cameras.length > 0 && (
        <div className={styles.stressPanel}>
          <h2>Multi-Camera Testing</h2>
          <div className={styles.stressFormRow}>
            <button
              className={styles.btnSecondary}
              disabled={allButtonsDisabled}
              onClick={() => startRunAllCameras(false)}
            >
              {runAllCamAllLabel}
            </button>
            <button
              className={styles.btnWarning}
              disabled={allButtonsDisabled}
              onClick={() => startRunAllCameras(true)}
            >
              {runAllCamMaxLabel}
            </button>
          </div>
        </div>
      )}

      {/* Video feed */}
      {showPanels && (
        <div className={styles.stressPanel}>
          <h3>
            {cameras.find((c) => c.id === selectedCamera)?.name ?? ''} Live
            Video Feed
          </h3>
          <div className={styles.stressVideoContainer}>
            <img
              src={stressVideoStreamUrl(selectedCamera!)}
              alt="Camera Stream"
              style={{ maxWidth: '100%', maxHeight: 400, display: 'block' }}
            />
          </div>
        </div>
      )}

      {/* Test control */}
      {showPanels && (
        <div className={styles.stressPanel}>
          <h2>Test Control</h2>
          <div className={styles.stressFormRow}>
            <button
              className={styles.btnPrimary}
              disabled={isRunning || allButtonsDisabled}
              onClick={doStartTest}
            >
              {isRunning ? 'Test Running...' : 'Start Test'}
            </button>
            <button
              className={styles.btnSecondary}
              disabled={allButtonsDisabled}
              onClick={startRunAllTests}
            >
              {runAllLabel}
            </button>
            <button
              className={styles.btnWarning}
              disabled={allButtonsDisabled}
              onClick={startRunMaxSpeedTests}
            >
              {runMaxLabel}
            </button>
            <button
              className={styles.btnDanger}
              disabled={!isRunning}
              onClick={handleAbort}
            >
              Abort Test
            </button>
          </div>

          {/* Progress */}
          {isRunning && (
            <>
              <div className={styles.progressBar} style={{ marginTop: 16 }}>
                <div
                  className={styles.progressBarFill}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <div className={styles.stressProgressText}>{progressText}</div>
            </>
          )}

          {/* PTZ display */}
          {ptzPosition && (
            <div className={styles.stressPtzDisplay}>
              <div className={styles.stressPtzAxis}>
                <div className={styles.stressPtzLabel}>Pan</div>
                <div className={styles.stressPtzValue}>
                  {ptzPosition.pan?.toFixed(1) ?? '--'}deg
                </div>
              </div>
              <div className={styles.stressPtzAxis}>
                <div className={styles.stressPtzLabel}>Tilt</div>
                <div className={styles.stressPtzValue}>
                  {ptzPosition.tilt?.toFixed(1) ?? '--'}deg
                </div>
              </div>
              <div className={styles.stressPtzAxis}>
                <div className={styles.stressPtzLabel}>Zoom</div>
                <div className={styles.stressPtzValue}>
                  {ptzPosition.zoom?.toFixed(1) ?? '--'}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Test configuration */}
      {showPanels && (
        <div className={styles.stressPanel}>
          <h2>Test Configuration</h2>

          {/* Presets */}
          <h3>Quick Presets</h3>
          <div className={styles.stressPresetsGrid}>
            {presets.map((p, i) => (
              <button
                key={p.name}
                className={`${styles.stressPresetBtn} ${selectedPresetIdx === i ? styles.stressPresetBtnSelected : ''}`}
                onClick={() => selectPreset(i)}
              >
                <div className={styles.presetName}>{p.name}</div>
                <div className={styles.presetDesc}>{p.description}</div>
              </button>
            ))}
          </div>

          {/* Manual config */}
          <h3 style={{ marginTop: 20 }}>Manual Configuration</h3>
          <div className={styles.stressFormRow}>
            <div className={styles.stressFormGroup}>
              <label>Test Type</label>
              <select
                className={styles.select}
                value={testType}
                onChange={(e) => setTestType(e.target.value)}
              >
                <option value="oscillation">Oscillation</option>
                <option value="random_step_accuracy">
                  Random Step Accuracy
                </option>
                <option value="full_range_sweep">Full Range Sweep</option>
                <option value="tilt_stress">Tilt Stress</option>
                <option value="combined_axis_load">Combined Axis Load</option>
                <option value="position_repeatability">
                  Position Repeatability
                </option>
                <option value="speed_test">Speed Test</option>
              </select>
            </div>
            <div className={styles.stressFormGroup}>
              <label>Repetitions</label>
              <input
                type="number"
                className={styles.numberInput}
                value={repetitions}
                min={1}
                max={100}
                onChange={(e) => setRepetitions(parseInt(e.target.value) || 1)}
              />
            </div>
          </div>

          {/* Pan axis */}
          <div className={styles.stressAxisConfig}>
            <h4>
              <input
                type="checkbox"
                checked={pan.enabled}
                onChange={(e) =>
                  setPan((p) => ({ ...p, enabled: e.target.checked }))
                }
              />{' '}
              Pan Axis Configuration
            </h4>
            <div className={styles.stressFormRow}>
              <div className={styles.stressFormGroup}>
                <label>Start (degrees)</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={pan.start}
                  step={0.1}
                  disabled={!pan.enabled}
                  onChange={(e) =>
                    setPan((p) => ({ ...p, start: parseFloat(e.target.value) || 0 }))
                  }
                />
              </div>
              <div className={styles.stressFormGroup}>
                <label>End (degrees)</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={pan.end}
                  step={0.1}
                  disabled={!pan.enabled}
                  onChange={(e) =>
                    setPan((p) => ({ ...p, end: parseFloat(e.target.value) || 0 }))
                  }
                />
              </div>
              <div className={styles.stressFormGroup}>
                <label>Step (degrees)</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={pan.step}
                  step={0.1}
                  disabled={!pan.enabled || pan.useRandomSteps}
                  onChange={(e) =>
                    setPan((p) => ({ ...p, step: parseFloat(e.target.value) || 0 }))
                  }
                />
              </div>
            </div>
            <div className={styles.stressFormRow}>
              <div className={styles.stressFormGroup}>
                <label>
                  <input
                    type="checkbox"
                    checked={pan.useRandomSteps}
                    disabled={!pan.enabled}
                    onChange={(e) =>
                      setPan((p) => ({
                        ...p,
                        useRandomSteps: e.target.checked,
                      }))
                    }
                  />{' '}
                  Use Random Steps
                </label>
              </div>
              <div className={styles.stressFormGroup}>
                <label>Min Step</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={pan.stepMin}
                  step={0.1}
                  disabled={!pan.enabled || !pan.useRandomSteps}
                  onChange={(e) =>
                    setPan((p) => ({
                      ...p,
                      stepMin: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
              <div className={styles.stressFormGroup}>
                <label>Max Step</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={pan.stepMax}
                  step={0.1}
                  disabled={!pan.enabled || !pan.useRandomSteps}
                  onChange={(e) =>
                    setPan((p) => ({
                      ...p,
                      stepMax: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
            </div>
          </div>

          {/* Tilt axis */}
          <div className={styles.stressAxisConfig}>
            <h4>
              <input
                type="checkbox"
                checked={tilt.enabled}
                onChange={(e) =>
                  setTilt((p) => ({ ...p, enabled: e.target.checked }))
                }
              />{' '}
              Tilt Axis Configuration
            </h4>
            <div className={styles.stressFormRow}>
              <div className={styles.stressFormGroup}>
                <label>Start (degrees)</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={tilt.start}
                  step={0.1}
                  disabled={!tilt.enabled}
                  onChange={(e) =>
                    setTilt((p) => ({
                      ...p,
                      start: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
              <div className={styles.stressFormGroup}>
                <label>End (degrees)</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={tilt.end}
                  step={0.1}
                  disabled={!tilt.enabled}
                  onChange={(e) =>
                    setTilt((p) => ({
                      ...p,
                      end: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
              <div className={styles.stressFormGroup}>
                <label>Step (degrees)</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={tilt.step}
                  step={0.1}
                  disabled={!tilt.enabled || tilt.useRandomSteps}
                  onChange={(e) =>
                    setTilt((p) => ({
                      ...p,
                      step: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
            </div>
            <div className={styles.stressFormRow}>
              <div className={styles.stressFormGroup}>
                <label>
                  <input
                    type="checkbox"
                    checked={tilt.useRandomSteps}
                    disabled={!tilt.enabled}
                    onChange={(e) =>
                      setTilt((p) => ({
                        ...p,
                        useRandomSteps: e.target.checked,
                      }))
                    }
                  />{' '}
                  Use Random Steps
                </label>
              </div>
              <div className={styles.stressFormGroup}>
                <label>Min Step</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={tilt.stepMin}
                  step={0.1}
                  disabled={!tilt.enabled || !tilt.useRandomSteps}
                  onChange={(e) =>
                    setTilt((p) => ({
                      ...p,
                      stepMin: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
              <div className={styles.stressFormGroup}>
                <label>Max Step</label>
                <input
                  type="number"
                  className={styles.numberInput}
                  value={tilt.stepMax}
                  step={0.1}
                  disabled={!tilt.enabled || !tilt.useRandomSteps}
                  onChange={(e) =>
                    setTilt((p) => ({
                      ...p,
                      stepMax: parseFloat(e.target.value) || 0,
                    }))
                  }
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test results */}
      {showResults && (
        <div className={styles.stressPanel}>
          <h2>Test Results</h2>
          <div className={styles.stressResultsPanel}>
            <div className={styles.stressResultItem}>
              <span className={styles.stressResultLabel}>Status</span>
              <span
                className={`${styles.stressResultValue} ${resultStatus === 'completed' ? styles.stressResultSuccess : styles.stressResultFail}`}
              >
                {resultStatus}
              </span>
            </div>
            <div className={styles.stressResultItem}>
              <span className={styles.stressResultLabel}>Position Match</span>
              <span
                className={`${styles.stressResultValue} ${resultPositionMatch === 'Yes' ? styles.stressResultSuccess : styles.stressResultFail}`}
              >
                {resultPositionMatch}
              </span>
            </div>
            <div className={styles.stressResultItem}>
              <span className={styles.stressResultLabel}>
                Position Error (Pan/Tilt/Zoom)
              </span>
              <span className={styles.stressResultValue}>
                {resultPositionError}
              </span>
            </div>
            <div className={styles.stressResultItem}>
              <span className={styles.stressResultLabel}>Total Duration</span>
              <span className={styles.stressResultValue}>{resultDuration}</span>
            </div>
            <div className={styles.stressResultItem}>
              <span className={styles.stressResultLabel}>
                Movements Completed
              </span>
              <span className={styles.stressResultValue}>
                {resultMovements}
              </span>
            </div>
          </div>

          {/* User evaluation */}
          <h3 style={{ marginTop: 20 }}>User Evaluation</h3>
          <div className={styles.stressEvalBtns}>
            <button
              className={`${styles.stressEvalBtn} ${styles.stressEvalGood} ${currentEvaluation === 'good' ? styles.stressEvalSelected : ''}`}
              disabled={!evalEnabled}
              onClick={() => setCurrentEvaluation('good')}
            >
              Good
            </button>
            <button
              className={`${styles.stressEvalBtn} ${styles.stressEvalNeedsImprovement} ${currentEvaluation === 'needs_improvement' ? styles.stressEvalSelected : ''}`}
              disabled={!evalEnabled}
              onClick={() => setCurrentEvaluation('needs_improvement')}
            >
              Needs Improvement
            </button>
            <button
              className={`${styles.stressEvalBtn} ${styles.stressEvalBad} ${currentEvaluation === 'bad' ? styles.stressEvalSelected : ''}`}
              disabled={!evalEnabled}
              onClick={() => setCurrentEvaluation('bad')}
            >
              Bad
            </button>
          </div>
          <textarea
            className={styles.stressNotesTextarea}
            placeholder="Optional notes about the test..."
            disabled={!evalEnabled}
            value={evalNotes}
            onChange={(e) => setEvalNotes(e.target.value)}
          />
          <button
            className={styles.btnPrimary}
            style={{ marginTop: 12, padding: '8px 16px', fontSize: 13 }}
            disabled={!evalEnabled || !currentEvaluation}
            onClick={saveEvaluation}
          >
            Save Evaluation
          </button>
        </div>
      )}

      {/* Past stress tests */}
      <div className={styles.stressPanel}>
        <h2>Past Stress Tests</h2>

        {sessionsLoading && (
          <div style={{ color: '#888', padding: 20, textAlign: 'center' }}>
            <span className={styles.spinner} /> Loading sessions...
          </div>
        )}

        {!sessionsLoading && sessions.length === 0 && (
          <div style={{ color: '#888', padding: 20, textAlign: 'center' }}>
            No stress test sessions found. Run a test to see results here.
          </div>
        )}

        {!sessionsLoading && sessions.length > 0 && (
          <table className={styles.stressSessionsTable}>
            <thead>
              <tr>
                <th>Date</th>
                <th>Tenant</th>
                <th>Camera</th>
                <th>Test Type</th>
                <th>Status</th>
                <th>Result</th>
                <th>Evaluation</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id}>
                  <td>{new Date(s.created_at).toLocaleString()}</td>
                  <td>{s.tenant_id ?? '--'}</td>
                  <td>{s.camera_name}</td>
                  <td>{s.test_type ?? '--'}</td>
                  <td>
                    <span
                      className={`${styles.stressStatusBadge} ${styles[`stressStatus_${s.status}`] ?? ''}`}
                    >
                      {s.status}
                    </span>
                  </td>
                  <td>
                    {s.result_success == null
                      ? '--'
                      : s.result_success
                        ? 'Pass'
                        : 'Fail'}
                  </td>
                  <td>
                    <span
                      className={`${styles.stressEvalBadge} ${styles[`stressEval_${s.user_evaluation}`] ?? ''}`}
                    >
                      {formatEvaluation(s.user_evaluation)}
                    </span>
                  </td>
                  <td className={styles.stressSessionActions}>
                    <button
                      className={styles.btnPrimary}
                      style={{ padding: '6px 12px', fontSize: 12 }}
                      onClick={() => viewSession(s.id)}
                    >
                      View
                    </button>
                    <button
                      className={styles.btnDanger}
                      style={{ padding: '6px 12px', fontSize: 12 }}
                      onClick={() => handleDeleteSession(s.id)}
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
    </div>
  );
}
