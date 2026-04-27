/**
 * Camera Diagnostic Tool page.
 *
 * Three tabs: Diagnostic Tests, Diagnostic History, and Stress Testing.
 * Mirrors the Django template at webapp/camera_diagnostic/templates/...
 */

import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import { useTenant } from '../contexts/TenantContext';
import DiagnosticTab from './camera-diagnostic/DiagnosticTab';
import HistoryTab from './camera-diagnostic/HistoryTab';
import StressTestTab from './camera-diagnostic/StressTestTab';
import type { CameraResult } from './camera-diagnostic/api';
import styles from './CameraDiagnosticPage.module.css';

type TabId = 'run-tests' | 'history' | 'stress-testing';

export default function CameraDiagnosticPage() {
  const { tenants, selectedTenantId } = useTenant();
  const [activeTab, setActiveTab] = useState<TabId>('run-tests');

  const handleViewSession = useCallback((_results: CameraResult[]) => {
    // TODO: could push results into DiagnosticTab for display
    setActiveTab('run-tests');
  }, []);

  const tenant = tenants?.find((t) => t.id === selectedTenantId);
  const tenantName = tenant?.name ?? selectedTenantId ?? '';

  if (!selectedTenantId) {
    return (
      <div className={styles.container}>
        <div className={styles.emptyState}>
          <p>Select a tenant to use the Camera Diagnostic tool.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <Link to="/" className={styles.backLink}>
        {'<-'} Back to Tools
      </Link>

      <div className={styles.header}>
        <h1>Camera Diagnostic Tool</h1>
        <p>
          Test camera connectivity: RTSP streaming, web interface, and PTZ API
        </p>
      </div>

      {/* Tabs */}
      <div className={styles.tabs}>
        {(
          [
            ['run-tests', 'Diagnostic Tests'],
            ['history', 'Diagnostic History'],
            ['stress-testing', 'Stress Testing'],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            className={`${styles.tab} ${activeTab === id ? styles.tabActive : ''}`}
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {activeTab === 'run-tests' && (
        <DiagnosticTab
          tenantId={selectedTenantId}
          tenantName={tenantName}
          key={`diag-${selectedTenantId}`}
        />
      )}

      {activeTab === 'history' && (
        <HistoryTab
          tenantId={selectedTenantId}
          tenantName={tenantName}
          onViewSession={handleViewSession}
          key={`hist-${selectedTenantId}`}
        />
      )}

      {activeTab === 'stress-testing' && (
        <StressTestTab
          tenantId={selectedTenantId}
          tenantName={tenantName}
          key={`stress-${selectedTenantId}`}
        />
      )}
    </div>
  );
}
