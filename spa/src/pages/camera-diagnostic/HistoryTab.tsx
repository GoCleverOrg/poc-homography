/**
 * Diagnostic History tab -- list/view/delete saved diagnostic sessions.
 */

import { useCallback, useEffect, useState } from 'react';
import type { CameraResult, DiagnosticSessionSummary } from './api';
import {
  fetchDiagnosticSessions,
  fetchDiagnosticSessionDetail,
  deleteDiagnosticSession,
} from './api';
import styles from '../CameraDiagnosticPage.module.css';

interface Props {
  tenantId: string;
  tenantName: string;
  /** Callback to switch to the Diagnostic tab and display a session's results. */
  onViewSession: (cameraResults: CameraResult[]) => void;
}

export default function HistoryTab({
  tenantId,
  tenantName,
  onViewSession,
}: Props) {
  const [sessions, setSessions] = useState<DiagnosticSessionSummary[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetchDiagnosticSessions(tenantId);
      if (res?.status === 'success' && res.data) {
        setSessions(res.data.sessions);
      }
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, [tenantId]);

  useEffect(() => {
    load();
  }, [load]);

  const viewSession = useCallback(
    async (sessionId: string) => {
      try {
        const res = await fetchDiagnosticSessionDetail(sessionId);
        if (res?.status === 'success' && res.data) {
          onViewSession(res.data.camera_results);
        }
      } catch {
        /* ignore */
      }
    },
    [onViewSession],
  );

  const handleDelete = useCallback(
    async (sessionId: string, e: React.MouseEvent) => {
      e.stopPropagation();
      if (!confirm('Are you sure you want to delete this session?')) return;
      try {
        const res = await deleteDiagnosticSession(sessionId);
        if (res?.status === 'success') load();
      } catch {
        /* ignore */
      }
    },
    [load],
  );

  return (
    <div>
      <div
        className={styles.selectorPanel}
        style={{ justifyContent: 'space-between' }}
      >
        <span className={styles.selectorLabel}>Tenant: {tenantName}</span>
        <button
          className={styles.btnPrimary}
          style={{ padding: '8px 16px', fontSize: 13 }}
          onClick={load}
        >
          Refresh
        </button>
      </div>

      {loading && (
        <div className={styles.emptyState}>
          <span className={styles.spinner} /> Loading sessions...
        </div>
      )}

      {!loading && sessions.length === 0 && (
        <div className={styles.emptyState}>
          <p>No diagnostic sessions found.</p>
          <p style={{ fontSize: 13 }}>Run tests to create a session.</p>
        </div>
      )}

      {!loading && sessions.length > 0 && (
        <div className={styles.sessionList}>
          {sessions.map((session) => {
            const summary = session.summary ?? {};
            const date = new Date(session.created_at).toLocaleString();
            return (
              <div
                key={session.id}
                className={styles.sessionItem}
                onClick={() => viewSession(session.id)}
              >
                <div className={styles.sessionInfo}>
                  <h3>{session.tenant_name || session.tenant_id}</h3>
                  <p>
                    {date} &bull; {summary.total_cameras ?? 0} cameras
                  </p>
                </div>
                <div className={styles.sessionStats}>
                  <span
                    className={`${styles.sessionStat} ${styles.sessionStatPass}`}
                  >
                    {summary.tests_passed ?? 0} passed
                  </span>
                  <span
                    className={`${styles.sessionStat} ${styles.sessionStatFail}`}
                  >
                    {summary.tests_failed ?? 0} failed
                  </span>
                  <button
                    className={styles.btnDanger}
                    onClick={(e) => handleDelete(session.id, e)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
