/**
 * Diagnostic Tests tab -- RTSP / WebUI / PTZ testing with result display,
 * video preview, snapshot capture, and session save.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Camera, CameraResult } from './api';
import {
  fetchCameras,
  fetchTestRtsp,
  fetchTestWebui,
  fetchTestPtz,
  saveDiagnosticSession,
  videoStreamUrl,
  snapshotUrl,
} from './api';
import styles from '../CameraDiagnosticPage.module.css';

/* ------------------------------------------------------------------ */
/* Types                                                              */
/* ------------------------------------------------------------------ */

type TestStatus = 'pending' | 'running' | 'pass' | 'fail';

interface CameraTestState {
  name: string;
  ip: string;
  rtsp: TestStatus;
  webui: TestStatus;
  ptz: TestStatus;
  rtspLogs: string;
  webuiLogs: string;
  ptzLogs: string;
  deviceInfo?: Record<string, unknown> | null;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

const TEST_TIMEOUTS: Record<string, number> = {
  rtsp: 30_000,
  webui: 60_000,
  ptz: 120_000,
};

function formatTestLogs(test: Record<string, unknown>): string {
  const lines: string[] = [];
  if (test.response_time_ms != null)
    lines.push(`Response time: ${test.response_time_ms}ms`);
  if (test.error_message) lines.push(`Error: ${test.error_message}`);
  if (test.details && typeof test.details === 'object') {
    for (const [key, value] of Object.entries(
      test.details as Record<string, unknown>,
    )) {
      lines.push(
        typeof value === 'object'
          ? `${key}: ${JSON.stringify(value)}`
          : `${key}: ${value}`,
      );
    }
  }
  return lines.join('\n') || 'No details available';
}

/* ------------------------------------------------------------------ */
/* Component                                                          */
/* ------------------------------------------------------------------ */

interface Props {
  tenantId: string;
  tenantName: string;
}

export default function DiagnosticTab({ tenantId, tenantName }: Props) {
  /* ---- state ---- */
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<
    Record<string, CameraTestState>
  >({});
  const [isRunning, setIsRunning] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [showVideo, setShowVideo] = useState(false);
  const [credentialsWarning, setCredentialsWarning] = useState(false);
  const [expandedCameras, setExpandedCameras] = useState<Set<string>>(
    new Set(),
  );

  const abortRef = useRef<AbortController | null>(null);

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

  /* ---- derived ---- */
  const passCount = Object.values(testResults).reduce(
    (n, r) =>
      n +
      (r.rtsp === 'pass' ? 1 : 0) +
      (r.webui === 'pass' ? 1 : 0) +
      (r.ptz === 'pass' ? 1 : 0),
    0,
  );
  const failCount = Object.values(testResults).reduce(
    (n, r) =>
      n +
      (r.rtsp === 'fail' ? 1 : 0) +
      (r.webui === 'fail' ? 1 : 0) +
      (r.ptz === 'fail' ? 1 : 0),
    0,
  );
  const pendingCount = Object.values(testResults).reduce(
    (n, r) =>
      n +
      (r.rtsp === 'pending' || r.rtsp === 'running' ? 1 : 0) +
      (r.webui === 'pending' || r.webui === 'running' ? 1 : 0) +
      (r.ptz === 'pending' || r.ptz === 'running' ? 1 : 0),
    0,
  );
  const total = passCount + failCount + pendingCount;
  const completedPct = total > 0 ? ((passCount + failCount) / total) * 100 : 0;

  /* ---- run a single test ---- */
  const runSingleTest = useCallback(
    async (
      cameraId: string,
      testType: 'rtsp' | 'webui' | 'ptz',
      parentSignal?: AbortSignal,
    ) => {
      // Mark running
      setTestResults((prev) => ({
        ...prev,
        [cameraId]: { ...prev[cameraId], [testType]: 'running' as TestStatus },
      }));

      const timeoutMs = TEST_TIMEOUTS[testType] ?? 60_000;
      const timeoutCtrl = new AbortController();
      const timer = setTimeout(() => timeoutCtrl.abort(), timeoutMs);

      if (parentSignal) {
        if (parentSignal.aborted) {
          clearTimeout(timer);
          timeoutCtrl.abort();
        } else {
          parentSignal.addEventListener(
            'abort',
            () => {
              clearTimeout(timer);
              timeoutCtrl.abort();
            },
            { once: true },
          );
        }
      }

      try {
        const fetcher =
          testType === 'rtsp'
            ? fetchTestRtsp
            : testType === 'webui'
              ? fetchTestWebui
              : fetchTestPtz;

        const result = await fetcher(cameraId, timeoutCtrl.signal);
        clearTimeout(timer);

        if (result?.status === 'success') {
          const d = (result.data ?? {}) as Record<string, unknown>;
          let status: TestStatus = 'pass';
          let logs = '';

          if (testType === 'rtsp') {
            const metrics = d.metrics as Record<string, unknown> | undefined;
            logs = `${(d.message as string) || 'Connected successfully'}\n`;
            if (metrics) {
              logs += `Latency: ${metrics.latency_ms}ms\n`;
              const res = metrics.resolution as Record<string, unknown>;
              if (res) logs += `Resolution: ${res.width}x${res.height}`;
            }
          } else if (testType === 'webui') {
            status = d.login_success ? 'pass' : 'fail';
            logs = `${(d.message as string) || 'Test completed'}\n`;
            if (d.login_attempted && !d.login_success) {
              logs += d.login_error
                ? `Error: ${d.login_error}\n`
                : 'Login failed silently\n';
            }
            const ptzControls = d.ptz_controls_found as unknown[];
            if (ptzControls?.length)
              logs += `PTZ controls found: ${ptzControls.length}\n`;
            if (d.screenshot_path) logs += `Screenshot: ${d.screenshot_path}`;
          } else {
            // ptz
            const tests = (d.tests ?? {}) as Record<
              string,
              Record<string, unknown>
            >;
            const allPassed = Object.values(tests).every(
              (t) => t.success !== false,
            );
            status = allPassed ? 'pass' : 'fail';
            const logLines: string[] = [];
            const initPos = d.initial_position as Record<string, number>;
            if (initPos) {
              logLines.push(
                `Initial: Pan=${initPos.pan?.toFixed(1)} Tilt=${initPos.tilt?.toFixed(1)} Zoom=${initPos.zoom?.toFixed(1)}`,
              );
            }
            for (const [name, test] of Object.entries(tests)) {
              const sym = test.success ? '✓' : '✗';
              logLines.push(
                `${sym} ${name}: ${test.success ? 'OK' : (test.error as string) || 'Failed'}`,
              );
            }
            if (d.position_restored) logLines.push('✓ Position restored');
            logs = logLines.join('\n');
          }

          setTestResults((prev) => ({
            ...prev,
            [cameraId]: {
              ...prev[cameraId],
              [testType]: status,
              [`${testType}Logs`]: logs,
            },
          }));
        } else {
          // error response
          const errResult = result as unknown as Record<string, unknown>;
          const msg = errResult?.message ?? 'Unknown error';
          if (errResult?.error_category === 'CREDENTIALS_NOT_SET')
            setCredentialsWarning(true);

          setTestResults((prev) => ({
            ...prev,
            [cameraId]: {
              ...prev[cameraId],
              [testType]: 'fail' as TestStatus,
              [`${testType}Logs`]: `Error: ${msg}`,
            },
          }));
        }
      } catch (err: unknown) {
        const e = err as Error;
        const msg =
          e.name === 'AbortError'
            ? parentSignal?.aborted
              ? 'Aborted by user'
              : `Request timed out after ${timeoutMs / 1000}s`
            : `Request failed: ${e.message}`;

        setTestResults((prev) => ({
          ...prev,
          [cameraId]: {
            ...prev[cameraId],
            [testType]: 'fail' as TestStatus,
            [`${testType}Logs`]: msg,
          },
        }));
      }
    },
    [],
  );

  /* ---- run all cameras ---- */
  const runAllTests = useCallback(async () => {
    if (isRunning) return;
    setIsRunning(true);
    setCredentialsWarning(false);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // Initialize results
    const init: Record<string, CameraTestState> = {};
    for (const cam of cameras) {
      init[cam.id] = {
        name: cam.name,
        ip: cam.ip ?? '',
        rtsp: 'pending',
        webui: 'pending',
        ptz: 'pending',
        rtspLogs: '',
        webuiLogs: '',
        ptzLogs: '',
      };
    }
    setTestResults(init);
    setShowSummary(true);
    setShowVideo(true);

    for (const cam of cameras) {
      if (ctrl.signal.aborted) break;
      // RTSP + WebUI concurrently, then PTZ
      await Promise.all([
        runSingleTest(cam.id, 'rtsp', ctrl.signal),
        runSingleTest(cam.id, 'webui', ctrl.signal),
      ]);
      if (ctrl.signal.aborted) break;
      await runSingleTest(cam.id, 'ptz', ctrl.signal);
    }

    // Save session
    try {
      // Read the latest testResults through a ref-like getter
      // (we capture current state snapshot after all tests)
      setTestResults((current) => {
        const cameraResultsList: CameraResult[] = cameras.map((cam) => {
          const r = current[cam.id];
          const buildTest = (
            type: 'rtsp' | 'webui' | 'ptz',
          ): {
            status: string;
            details: Record<string, unknown>;
            error_message?: string;
          } => {
            const status =
              r[type] === 'pass'
                ? 'pass'
                : r[type] === 'fail'
                  ? 'fail'
                  : 'pending';
            const logsKey = `${type}Logs` as keyof CameraTestState;
            const logs = (r[logsKey] as string) || '';
            const result: {
              status: string;
              details: Record<string, unknown>;
              error_message?: string;
            } = { status, details: { logs } };
            if (status === 'fail') result.error_message = logs || 'Test failed';
            return result;
          };
          return {
            camera_id: cam.id,
            camera_name: r.name || cam.name,
            camera_ip: r.ip || cam.ip || '',
            rtsp_test: buildTest('rtsp'),
            webui_test: buildTest('webui'),
            ptz_test: buildTest('ptz'),
          };
        });
        saveDiagnosticSession(tenantId, cameraResultsList).catch(() => {
          /* best-effort */
        });
        return current; // no mutation
      });
    } catch {
      /* best-effort */
    }

    abortRef.current = null;
    setIsRunning(false);
  }, [isRunning, cameras, tenantId, runSingleTest]);

  /* ---- run single camera ---- */
  const runSingleCameraTest = useCallback(
    async (testType: 'rtsp' | 'webui' | 'ptz') => {
      if (isRunning || !selectedCamera) return;
      setIsRunning(true);

      const cam = cameras.find((c) => c.id === selectedCamera);
      setTestResults((prev) => ({
        ...prev,
        [selectedCamera]: prev[selectedCamera] ?? {
          name: cam?.name ?? selectedCamera,
          ip: cam?.ip ?? '',
          rtsp: 'pending',
          webui: 'pending',
          ptz: 'pending',
          rtspLogs: '',
          webuiLogs: '',
          ptzLogs: '',
        },
      }));
      setShowSummary(true);

      await runSingleTest(selectedCamera, testType);
      setIsRunning(false);
    },
    [isRunning, selectedCamera, cameras, runSingleTest],
  );

  const runAllSingleCamera = useCallback(async () => {
    if (isRunning || !selectedCamera) return;
    setIsRunning(true);

    const cam = cameras.find((c) => c.id === selectedCamera);
    const init: Record<string, CameraTestState> = {};
    init[selectedCamera] = {
      name: cam?.name ?? selectedCamera,
      ip: cam?.ip ?? '',
      rtsp: 'pending',
      webui: 'pending',
      ptz: 'pending',
      rtspLogs: '',
      webuiLogs: '',
      ptzLogs: '',
    };
    setTestResults(init);
    setShowSummary(true);

    await Promise.all([
      runSingleTest(selectedCamera, 'rtsp'),
      runSingleTest(selectedCamera, 'webui'),
    ]);
    await runSingleTest(selectedCamera, 'ptz');
    setIsRunning(false);
  }, [isRunning, selectedCamera, cameras, runSingleTest]);

  /* ---- abort ---- */
  const handleAbort = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  /* ---- snapshot ---- */
  const captureSnapshot = useCallback(async () => {
    if (!selectedCamera) return;
    try {
      const resp = await fetch(snapshotUrl(selectedCamera));
      if (resp.ok) {
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedCamera}_snapshot_${Date.now()}.jpg`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch {
      /* ignore */
    }
  }, [selectedCamera]);

  /* ---- camera select change ---- */
  const onCameraChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value || null;
      setSelectedCamera(val);
      setShowVideo(!!val);
      if (!val) setShowSummary(false);
    },
    [],
  );

  /* ---- toggle expand ---- */
  const toggleExpand = useCallback((cameraId: string) => {
    setExpandedCameras((prev) => {
      const next = new Set(prev);
      if (next.has(cameraId)) next.delete(cameraId);
      else next.add(cameraId);
      return next;
    });
  }, []);

  /* ---- populate results from session ---- */
  const loadFromSession = useCallback(
    (cameraResults: CameraResult[]) => {
      const results: Record<string, CameraTestState> = {};
      for (const cr of cameraResults) {
        results[cr.camera_id] = {
          name: cr.camera_name,
          ip: cr.camera_ip,
          rtsp: cr.rtsp_test?.status === 'pass' ? 'pass' : cr.rtsp_test ? 'fail' : 'pending',
          webui: cr.webui_test?.status === 'pass' ? 'pass' : cr.webui_test ? 'fail' : 'pending',
          ptz: cr.ptz_test?.status === 'pass' ? 'pass' : cr.ptz_test ? 'fail' : 'pending',
          rtspLogs: cr.rtsp_test ? formatTestLogs(cr.rtsp_test as unknown as Record<string, unknown>) : '',
          webuiLogs: cr.webui_test ? formatTestLogs(cr.webui_test as unknown as Record<string, unknown>) : '',
          ptzLogs: cr.ptz_test ? formatTestLogs(cr.ptz_test as unknown as Record<string, unknown>) : '',
          deviceInfo: cr.device_info,
        };
      }
      setTestResults(results);
      setShowSummary(true);
    },
    [],
  );

  // Expose loadFromSession via a ref for the history tab
  const loadFromSessionRef = useRef(loadFromSession);
  loadFromSessionRef.current = loadFromSession;

  /* ---- badge class ---- */
  const badgeCls = (status: TestStatus) =>
    `${styles.testBadge} ${status === 'pass' ? styles.testBadgePass : status === 'fail' ? styles.testBadgeFail : status === 'running' ? styles.testBadgeRunning : styles.testBadgePending}`;

  /* ---- render ---- */
  return (
    <div>
      {/* Credentials warning */}
      {credentialsWarning && (
        <div className={styles.credentialsWarning}>
          <strong>Camera Credentials Not Set</strong>
          <p>Set environment variables before running tests:</p>
          <code>
            export CAMERA_USERNAME=admin
            <br />
            export CAMERA_PASSWORD=your_password
          </code>
        </div>
      )}

      {/* Camera selector */}
      <div className={styles.selectorPanel}>
        <span className={styles.selectorLabel}>Tenant: {tenantName}</span>
        <label htmlFor="diag-camera-select">Camera:</label>
        <select
          id="diag-camera-select"
          className={styles.select}
          disabled={cameras.length === 0}
          value={selectedCamera ?? ''}
          onChange={onCameraChange}
        >
          <option value="">-- All cameras --</option>
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name} ({c.ip})
            </option>
          ))}
        </select>

        <button
          className={styles.btnPrimary}
          disabled={cameras.length === 0 || isRunning}
          onClick={runAllTests}
        >
          {isRunning ? (
            <>
              <span className={styles.spinner} /> Running...
            </>
          ) : (
            `Run All Tests on ${cameras.length} Camera(s)`
          )}
        </button>

        {isRunning && (
          <button className={styles.btnWarning} onClick={handleAbort}>
            Abort
          </button>
        )}
      </div>

      {/* Single camera actions */}
      {selectedCamera && (
        <div className={styles.selectorPanel}>
          <span className={styles.selectorLabel}>
            {cameras.find((c) => c.id === selectedCamera)?.name ?? selectedCamera}
          </span>
          <button
            className={styles.btnSecondary}
            disabled={isRunning}
            onClick={() => runSingleCameraTest('rtsp')}
          >
            Test RTSP
          </button>
          <button
            className={styles.btnSecondary}
            disabled={isRunning}
            onClick={() => runSingleCameraTest('webui')}
          >
            Test WebUI
          </button>
          <button
            className={styles.btnSecondary}
            disabled={isRunning}
            onClick={() => runSingleCameraTest('ptz')}
          >
            Test PTZ
          </button>
          <button
            className={styles.btnPrimary}
            disabled={isRunning}
            onClick={runAllSingleCamera}
          >
            Run All Tests
          </button>
        </div>
      )}

      {/* Summary report */}
      {showSummary && (
        <div className={styles.summaryReport}>
          <div className={styles.summaryHeader}>
            <h2>Test Results Summary</h2>
            <div className={styles.summaryStats}>
              <div className={`${styles.statItem} ${styles.statPass}`}>
                <span>{'✓'}</span>
                <span>{passCount}</span> Passed
              </div>
              <div className={`${styles.statItem} ${styles.statFail}`}>
                <span>{'✗'}</span>
                <span>{failCount}</span> Failed
              </div>
              <div className={`${styles.statItem} ${styles.statPending}`}>
                <span>{'⌛'}</span>
                <span>{pendingCount}</span> Pending
              </div>
            </div>
          </div>
          <div className={styles.progressBar}>
            <div
              className={styles.progressBarFill}
              style={{ width: `${completedPct}%` }}
            />
          </div>

          {/* Camera results */}
          <div className={styles.cameraResults}>
            {Object.entries(testResults).map(([camId, result]) => {
              const isExpanded = expandedCameras.has(camId);

              // Device info lines
              const deviceLines: string[] = [];
              if (result.deviceInfo) {
                const di = result.deviceInfo as Record<string, string>;
                if (di.model) deviceLines.push(`Model: ${di.model}`);
                if (di.serial_number) deviceLines.push(`Serial: ${di.serial_number}`);
                if (di.mac_address) deviceLines.push(`MAC: ${di.mac_address}`);
                if (di.firmware_version) deviceLines.push(`Firmware: ${di.firmware_version}`);
                if (di.device_name) deviceLines.push(`Device Name: ${di.device_name}`);
              }

              return (
                <div
                  key={camId}
                  className={styles.cameraResultItem}
                >
                  <div
                    className={styles.cameraResultHeader}
                    onClick={() => toggleExpand(camId)}
                  >
                    <div className={styles.cameraResultTitle}>
                      <h3>
                        {result.name} ({result.ip})
                      </h3>
                      <div className={styles.testBadges}>
                        <span className={badgeCls(result.rtsp)}>
                          RTSP: {result.rtsp.toUpperCase()}
                        </span>
                        <span className={badgeCls(result.webui)}>
                          WebUI: {result.webui.toUpperCase()}
                        </span>
                        <span className={badgeCls(result.ptz)}>
                          PTZ: {result.ptz.toUpperCase()}
                        </span>
                      </div>
                    </div>
                    <span
                      className={`${styles.expandIcon} ${isExpanded ? styles.expandIconRotated : ''}`}
                    >
                      {'▼'}
                    </span>
                  </div>

                  {isExpanded && (
                    <div className={styles.cameraResultDetails}>
                      {deviceLines.length > 0 && (
                        <div className={styles.testSectionMini}>
                          <h4>Camera Hardware Info</h4>
                          <div className={styles.logOutput}>
                            {deviceLines.join('\n')}
                          </div>
                        </div>
                      )}
                      <div className={styles.testSectionMini}>
                        <h4>
                          <span className={badgeCls(result.rtsp)}>
                            {result.rtsp.toUpperCase()}
                          </span>{' '}
                          RTSP Streaming
                        </h4>
                        <div className={styles.logOutput}>
                          {result.rtspLogs || 'No logs yet'}
                        </div>
                      </div>
                      <div className={styles.testSectionMini}>
                        <h4>
                          <span className={badgeCls(result.webui)}>
                            {result.webui.toUpperCase()}
                          </span>{' '}
                          Web Interface
                        </h4>
                        <div className={styles.logOutput}>
                          {result.webuiLogs || 'No logs yet'}
                        </div>
                      </div>
                      <div className={styles.testSectionMini}>
                        <h4>
                          <span className={badgeCls(result.ptz)}>
                            {result.ptz.toUpperCase()}
                          </span>{' '}
                          PTZ API
                        </h4>
                        <div className={styles.logOutput}>
                          {result.ptzLogs || 'No logs yet'}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Video preview */}
      {showVideo && selectedCamera && (
        <div className={styles.videoPreviewContainer}>
          <div className={styles.videoPreviewHeader}>
            <h3>
              {cameras.find((c) => c.id === selectedCamera)?.name ?? 'Camera'}{' '}
              Preview
            </h3>
            <button
              className={styles.btnPrimary}
              style={{ padding: '8px 16px', fontSize: 13 }}
              onClick={captureSnapshot}
            >
              Capture Snapshot
            </button>
          </div>
          <div className={styles.videoContainer}>
            <img
              src={videoStreamUrl(selectedCamera)}
              alt="RTSP Stream"
              style={{ maxWidth: '100%', display: 'block' }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
