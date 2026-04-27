/**
 * Camera Diagnostic API helpers.
 *
 * Wraps the openapi-fetch client for all /camera-diagnostic/ endpoints and
 * provides a few plain-fetch helpers for streaming / binary endpoints that
 * openapi-fetch cannot handle (MJPEG streams, JPEG snapshots).
 */

import client from '../../api/client';

/* ------------------------------------------------------------------ */
/* Generic JSON wrapper returned by almost every endpoint              */
/* ------------------------------------------------------------------ */

export interface ApiEnvelope<T = unknown> {
  status: 'success' | 'error';
  data?: T;
  message?: string;
  error_category?: string;
}

/* ------------------------------------------------------------------ */
/* Shared domain types                                                */
/* ------------------------------------------------------------------ */

export interface Camera {
  id: string;
  name: string;
  ip: string | null;
  tenant_id?: string | null;
}

export interface Tenant {
  id: string;
  name: string;
  description?: string | null;
}

export interface TestResult {
  test_type?: string;
  status: string;
  response_time_ms?: number | null;
  error_message?: string | null;
  error_category?: string | null;
  details?: Record<string, unknown>;
}

export interface CameraResult {
  camera_id: string;
  camera_name: string;
  camera_ip: string;
  device_info?: Record<string, unknown> | null;
  rtsp_test?: TestResult | null;
  webui_test?: TestResult | null;
  ptz_test?: TestResult | null;
}

export interface DiagnosticSessionSummary {
  id: string;
  created_at: string;
  completed_at?: string | null;
  status: string;
  tenant_id: string;
  tenant_name: string;
  summary?: {
    total_cameras?: number;
    tests_passed?: number;
    tests_failed?: number;
    tests_pending?: number;
  };
}

export interface DiagnosticSessionDetail {
  id: string;
  created_at: string;
  completed_at?: string | null;
  status: string;
  tenant_id: string;
  tenant_name: string;
  camera_results: CameraResult[];
}

export interface StressPreset {
  name: string;
  description: string;
  test_type: string;
  repetitions: number;
  max_speed: boolean;
  pan_config: AxisConfig | null;
  tilt_config: AxisConfig | null;
  zoom_config: AxisConfig | null;
}

export interface AxisConfig {
  axis: string;
  start: number;
  end: number;
  step: number;
  use_random_steps: boolean;
  step_min: number;
  step_max: number;
}

export interface StressSessionSummary {
  id: string;
  created_at: string;
  completed_at?: string | null;
  status: string;
  tenant_id?: string;
  camera_id?: string;
  camera_name: string;
  test_type?: string | null;
  user_evaluation: string;
  result_success?: boolean | null;
}

export interface StressProgress {
  session_id: string;
  status: string;
  current_repetition: number;
  total_repetitions: number;
  current_movement?: number;
  total_movements?: number;
  message: string;
  completed?: boolean;
  current_position?: {
    pan?: number;
    tilt?: number;
    zoom?: number;
  };
}

export interface StressSessionDetail {
  session: {
    id: string;
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
    config?: {
      test_type: string;
      repetitions: number;
    };
  };
}

/* ------------------------------------------------------------------ */
/* Diagnostic tab endpoints                                           */
/*                                                                    */
/* NOTE: The OpenAPI spec types don't reflect the envelope pattern    */
/* used by camera-diagnostic endpoints, so `data` is typed as the    */
/* raw response schema (or undefined). We use `(data ?? {}) as`      */
/* to assert the actual envelope shape. This is a single-step cast   */
/* (data is already `unknown`-ish from the client) and is preferable */
/* to the double-cast `as unknown as` anti-pattern.                  */
/* ------------------------------------------------------------------ */

export async function fetchCameras(tenantId: string) {
  const { data } = await client.GET('/camera-diagnostic/api/cameras/', {
    params: { query: { tenant_id: tenantId } },
  });
  return (data ?? {}) as ApiEnvelope<{ cameras: Camera[] }>;
}

export async function fetchTestRtsp(cameraName: string, signal?: AbortSignal) {
  const { data } = await client.GET(
    '/camera-diagnostic/api/test-rtsp/{camera_name}/',
    { params: { path: { camera_name: cameraName } }, signal },
  );
  return (data ?? {}) as ApiEnvelope;
}

export async function fetchTestWebui(
  cameraName: string,
  signal?: AbortSignal,
) {
  const { data } = await client.GET(
    '/camera-diagnostic/api/test-webui/{camera_name}/',
    { params: { path: { camera_name: cameraName } }, signal },
  );
  return (data ?? {}) as ApiEnvelope;
}

export async function fetchTestPtz(cameraName: string, signal?: AbortSignal) {
  const { data } = await client.GET(
    '/camera-diagnostic/api/test-ptz/{camera_name}/',
    { params: { path: { camera_name: cameraName } }, signal },
  );
  return (data ?? {}) as ApiEnvelope;
}

export async function saveDiagnosticSession(
  tenantId: string,
  cameraResults: CameraResult[],
) {
  const { data } = await client.POST(
    '/camera-diagnostic/api/diagnostic/save/',
    {
      // `as never` — the OpenAPI spec doesn't define a request body for these
      // endpoints, so the generated types reject any body. The cast is required
      // until the spec is updated.
      body: {
        tenant_id: tenantId,
        camera_results: cameraResults,
      } as never,
    },
  );
  return (data ?? {}) as ApiEnvelope;
}

export async function fetchDiagnosticSessions(tenantId: string) {
  const { data } = await client.GET(
    '/camera-diagnostic/api/diagnostic/sessions/',
    { params: { query: { tenant_id: tenantId } } },
  );
  return (data ?? {}) as ApiEnvelope<{
    sessions: DiagnosticSessionSummary[];
    total: number;
  }>;
}

export async function fetchDiagnosticSessionDetail(sessionId: string) {
  const { data } = await client.GET(
    '/camera-diagnostic/api/diagnostic/sessions/{session_id}/',
    { params: { path: { session_id: sessionId } } },
  );
  return (data ?? {}) as ApiEnvelope<DiagnosticSessionDetail>;
}

export async function deleteDiagnosticSession(sessionId: string) {
  const { data } = await client.POST(
    '/camera-diagnostic/api/diagnostic/sessions/{session_id}/delete/',
    { params: { path: { session_id: sessionId } } },
  );
  return (data ?? {}) as ApiEnvelope;
}

/* ------------------------------------------------------------------ */
/* Stress testing endpoints                                           */
/* ------------------------------------------------------------------ */

export async function fetchStressPresets() {
  const { data } = await client.GET(
    '/camera-diagnostic/api/stress-test/presets/',
  );
  return (data ?? {}) as ApiEnvelope<{ presets: StressPreset[] }>;
}

export async function startStressTest(body: {
  tenant_id: string;
  camera_id: string;
  preset_name?: string;
  test_type?: string;
  pan_config?: Record<string, unknown>;
  tilt_config?: Record<string, unknown>;
  repetitions?: number;
  max_speed?: boolean;
}) {
  const { data } = await client.POST(
    '/camera-diagnostic/api/stress-test/start/',
    // `as never` — see saveDiagnosticSession for rationale.
    { body: body as never },
  );
  return (data ?? {}) as ApiEnvelope<{
    session_id: string;
    message: string;
  }>;
}

export async function fetchStressStatus(sessionId: string) {
  const { data } = await client.GET(
    '/camera-diagnostic/api/stress-test/status/{session_id}/',
    { params: { path: { session_id: sessionId } } },
  );
  return (data ?? {}) as ApiEnvelope<StressProgress>;
}

export async function abortStressTest(sessionId: string) {
  const { data } = await client.POST(
    '/camera-diagnostic/api/stress-test/abort/{session_id}/',
    { params: { path: { session_id: sessionId } } },
  );
  return (data ?? {}) as ApiEnvelope;
}

export async function fetchStressSessions(
  tenantId?: string,
  cameraId?: string,
) {
  const query: Record<string, string> = {};
  if (tenantId) query.tenant_id = tenantId;
  if (cameraId) query.camera_id = cameraId;
  const { data } = await client.GET(
    '/camera-diagnostic/api/stress-test/sessions/',
    { params: { query } },
  );
  return (data ?? {}) as ApiEnvelope<{
    sessions: StressSessionSummary[];
    total: number;
  }>;
}

export async function fetchStressSessionDetail(sessionId: string) {
  const { data } = await client.GET(
    '/camera-diagnostic/api/stress-test/sessions/{session_id}/',
    { params: { path: { session_id: sessionId } } },
  );
  return (data ?? {}) as ApiEnvelope<StressSessionDetail>;
}

export async function submitStressEvaluation(
  sessionId: string,
  evaluation: string,
  notes: string,
) {
  const { data } = await client.POST(
    '/camera-diagnostic/api/stress-test/sessions/{session_id}/evaluate/',
    {
      params: { path: { session_id: sessionId } },
      // `as never` — see saveDiagnosticSession for rationale.
      body: { evaluation, notes } as never,
    },
  );
  return (data ?? {}) as ApiEnvelope;
}

export async function deleteStressSession(sessionId: string) {
  const { data } = await client.POST(
    '/camera-diagnostic/api/stress-test/sessions/{session_id}/delete/',
    { params: { path: { session_id: sessionId } } },
  );
  return (data ?? {}) as ApiEnvelope;
}

/* ------------------------------------------------------------------ */
/* URL builders (for MJPEG / snapshot binary endpoints)               */
/* ------------------------------------------------------------------ */

export function videoStreamUrl(cameraName: string): string {
  return `/camera-diagnostic/api/video-stream/${encodeURIComponent(cameraName)}/?t=${Date.now()}`;
}

export function stressVideoStreamUrl(cameraId: string): string {
  return `/camera-diagnostic/api/stress-test/video-stream/${encodeURIComponent(cameraId)}/?t=${Date.now()}`;
}

export function snapshotUrl(cameraName: string): string {
  return `/camera-diagnostic/api/capture-snapshot/${encodeURIComponent(cameraName)}/`;
}
