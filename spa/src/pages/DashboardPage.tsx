import { Link } from 'react-router-dom';
import { useTenant } from '../contexts/TenantContext';
import { useTenantMaps } from '../hooks/useTenantMaps';
import styles from './DashboardPage.module.css';

interface ToolCardDef {
  to: string;
  title: string;
  description: string;
  badge?: string;
  /** When true the card is always enabled regardless of map availability. */
  alwaysEnabled?: boolean;
}

const TOOL_CARDS: ToolCardDef[] = [
  {
    to: '/click-to-gps',
    title: 'Click-to-GPS',
    badge: 'Primary',
    description:
      'Click anywhere on a camera frame to estimate its GPS coordinate. Confidence-coloured markers, last-5 history, copy-to-clipboard.',
  },
  {
    to: '/point-picker',
    title: 'Map Points Capture Tool',
    badge: 'Primary',
    description:
      'Capture and annotate map points on high-resolution site reference images. Supports per-site selection, tiled image viewing, and coordinate export.',
  },
  {
    to: '/homography-precision',
    title: 'Homography Precision Tool',
    description:
      'Visualize and analyze homography transformation precision across camera frames. Compare predicted vs. actual GCP locations.',
  },
  {
    to: '/line-picker',
    title: 'Line Picker',
    description:
      'Select and annotate lines on camera frames for calibration and analysis.',
  },
  {
    to: '/camera-annotator',
    title: 'Camera Frame Annotator',
    description:
      'Annotate camera frames with GCP references. Generate YAML annotations for test cases.',
  },
  {
    to: '/camera-line-annotator',
    title: 'Camera Line Annotator',
    badge: 'New',
    description:
      'Annotate lines on camera frames for line-based homography testing. Create test cases linking map lines to camera frame lines.',
  },
  {
    to: '/camera-diagnostic',
    title: 'Camera Diagnostic Tool',
    badge: 'Diagnostic',
    alwaysEnabled: true,
    description:
      'Test camera connectivity, RTSP streaming, web interface, and PTZ control functionality. Verify Hikvision camera health and troubleshoot issues.',
  },
  {
    to: '/camera-evaluation',
    title: 'Camera Evaluation Tool',
    badge: 'Evaluation',
    alwaysEnabled: true,
    description:
      'Survey camera surroundings with PTZ control and run stress tests to validate equipment installation. Two tabs: Survey (PTZ sweep with screenshots) and Stress Testing (PTZ reliability tests).',
  },
  {
    to: '/lens-calibration',
    title: 'Lens Calibration Tool',
    badge: 'Calibration',
    alwaysEnabled: true,
    description:
      'Calculate lens distortion coefficients (k1, k2, k3, p1, p2) using detected parking lines. Optimizes for line straightness and saves calibration for homography correction.',
  },
  {
    to: '/distortion-validator',
    title: 'Distortion Validator',
    badge: 'Validation',
    alwaysEnabled: true,
    description:
      'Validate lens distortion calibration by viewing undistorted images side-by-side. Draw lines on undistorted images to verify straightness and calibration quality.',
  },
  {
    to: '/clean-plate-gallery',
    title: 'Clean-Plate Gallery',
    badge: 'New',
    alwaysEnabled: true,
    description:
      "Browse maglor's captured clean-plate floor images by run. Responsive thumbnail grid (imgproxy-served) with per-frame metadata; click a thumbnail for the full image and pose/optics details.",
  },
];

export default function DashboardPage() {
  const { selectedTenantId } = useTenant();
  const { mapIds, loading } = useTenantMaps();

  const hasMaps = Array.isArray(mapIds) && mapIds.length > 0;

  return (
    <div className={styles.hero}>
      <h1 className={styles.heroTitle}>Homography GCP Tools</h1>
      <p className={styles.subtitle}>
        Ground Control Point management and visualization for camera homography
      </p>

      {/* Maps info bar */}
      {loading && (
        <div className={styles.mapsInfo}>
          <span className={styles.mapsInfoLabel}>Maps:</span> Loading...
        </div>
      )}
      {!loading && hasMaps && (
        <div className={styles.mapsInfo}>
          <span className={styles.mapsInfoLabel}>Maps:</span>{' '}
          {mapIds.join(', ')}
        </div>
      )}

      {/* No-map warning */}
      {!loading && !hasMaps && selectedTenantId && (
        <div className={styles.noMapWarning}>
          <strong>
            No map data available for tenant &ldquo;{selectedTenantId}&rdquo;
          </strong>
          <p>
            Map-dependent tools require GCP data to be configured for this
            tenant&apos;s map. Contact an administrator to set up map and GCP
            data.
          </p>
        </div>
      )}

      {/* Tool card grid */}
      <div className={styles.toolGrid}>
        {TOOL_CARDS.map((card) => {
          const disabled = !card.alwaysEnabled && !hasMaps;
          const className = [styles.toolCard, disabled ? styles.disabled : '']
            .filter(Boolean)
            .join(' ');

          const content = (
            <>
              <div className={styles.toolTitle}>
                {card.title}
                {card.badge && (
                  <span className={styles.badge}>{card.badge}</span>
                )}
              </div>
              <div className={styles.toolDescription}>{card.description}</div>
            </>
          );

          return disabled ? (
            <div
              key={card.to}
              className={className}
              aria-disabled
            >
              {content}
            </div>
          ) : (
            <Link
              key={card.to}
              to={card.to}
              className={className}
            >
              {content}
            </Link>
          );
        })}
      </div>

      {/* Footer */}
      <div className={styles.footer}>
        <p>POC Homography Project</p>
        <p className={styles.footerSub}>Built with React</p>
      </div>
    </div>
  );
}
