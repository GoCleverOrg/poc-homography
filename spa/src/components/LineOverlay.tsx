import { useRef, useEffect, useState, useCallback } from 'react';
import OpenSeadragon from 'openseadragon';

// ------------------------------------------------------------------ types

export interface LineData {
  /** Stable identifier for the line (e.g. "L1"). */
  id: string;
  /** Image-pixel coordinates of the line's vertices (at least 2). */
  points: Array<{ x: number; y: number }>;
  /** Stroke colour (any CSS colour string). */
  color: string;
  /** Whether this line is currently selected. */
  selected?: boolean;
}

export interface LineOverlayProps {
  /** The OSD viewer instance (pass `null` until the viewer is ready). */
  viewer: OpenSeadragon.Viewer | null;
  /** Lines to render, specified in image-pixel coordinates. */
  lines: LineData[];
  /** Base stroke width in CSS pixels (default 3). */
  strokeWidth?: number;
  /** Stroke width used for selected lines (default `strokeWidth + 1`). */
  selectedStrokeWidth?: number;
  /** Callback when a line is clicked. */
  onLineClick?: (lineId: string) => void;
}

// -------------------------------------------------------- pixel conversion

interface ScreenPoint {
  x: number;
  y: number;
}

/**
 * Convert an image-pixel coordinate to a viewer-element pixel coordinate
 * suitable for an absolutely-positioned SVG overlay.
 */
function toScreenPixel(
  viewer: OpenSeadragon.Viewer,
  imgX: number,
  imgY: number,
): ScreenPoint {
  const vp = viewer.viewport.imageToViewportCoordinates(imgX, imgY);
  const px = viewer.viewport.pixelFromPoint(vp);
  return { x: px.x, y: px.y };
}

// -------------------------------------------------------------- component

/**
 * SVG overlay that draws lines on top of the OSD viewer.
 *
 * The SVG element is absolutely positioned over the viewer container and
 * re-renders whenever the viewport changes (pan / zoom / resize).  Lines
 * are specified in image-pixel coordinates; the component handles the
 * conversion to screen pixels.
 *
 * Usage:
 * ```tsx
 * <div style={{ position: 'relative', width: '100%', height: '100%' }}>
 *   <OpenSeadragonViewer tileSource={ts} onViewerReady={setViewer} />
 *   <LineOverlay viewer={viewer} lines={lines} onLineClick={selectLine} />
 * </div>
 * ```
 */
export default function LineOverlay({
  viewer,
  lines,
  strokeWidth = 3,
  selectedStrokeWidth,
  onLineClick,
}: LineOverlayProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  // We keep a simple counter that increments whenever the viewport moves,
  // which triggers a React re-render with up-to-date screen coordinates.
  const [, setTick] = useState(0);
  const bump = useCallback(() => setTick((t) => t + 1), []);

  // Subscribe to viewport changes (pan, zoom, resize, animation frames)
  useEffect(() => {
    if (!viewer) return;

    // 'animation' fires on every frame while animating; 'animation-finish'
    // catches the final resting position.  'resize' handles container
    // dimension changes.
    const events = [
      'animation',
      'animation-finish',
      'resize',
      'open',
    ] as const;

    for (const name of events) {
      viewer.addHandler(name, bump);
    }

    return () => {
      for (const name of events) {
        viewer.removeHandler(name, bump);
      }
    };
  }, [viewer, bump]);

  // ---- nothing to draw yet ----
  if (!viewer || lines.length === 0) return null;

  const selWidth = selectedStrokeWidth ?? strokeWidth + 1;

  return (
    <svg
      ref={svgRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 10,
      }}
    >
      {lines.map((line) => {
        if (line.points.length < 2) return null;

        const isSelected = line.selected ?? false;
        const sw = isSelected ? selWidth : strokeWidth;

        // Two-point line: render a <line> for simplicity and perf.
        if (line.points.length === 2) {
          const [p0, p1] = line.points;
          const a = toScreenPixel(viewer, p0.x, p0.y);
          const b = toScreenPixel(viewer, p1.x, p1.y);

          return (
            <line
              key={line.id}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={isSelected ? '#ff9800' : line.color}
              strokeWidth={sw}
              strokeLinecap="round"
              style={{
                pointerEvents: 'stroke',
                cursor: onLineClick ? 'pointer' : undefined,
              }}
              onClick={
                onLineClick
                  ? (e) => {
                      e.stopPropagation();
                      onLineClick(line.id);
                    }
                  : undefined
              }
            />
          );
        }

        // Multi-point line: render a <polyline>.
        const pointsAttr = line.points
          .map((p) => {
            const s = toScreenPixel(viewer, p.x, p.y);
            return `${s.x},${s.y}`;
          })
          .join(' ');

        return (
          <polyline
            key={line.id}
            points={pointsAttr}
            fill="none"
            stroke={isSelected ? '#ff9800' : line.color}
            strokeWidth={sw}
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{
              pointerEvents: 'stroke',
              cursor: onLineClick ? 'pointer' : undefined,
            }}
            onClick={
              onLineClick
                ? (e) => {
                    e.stopPropagation();
                    onLineClick(line.id);
                  }
                : undefined
            }
          />
        );
      })}
    </svg>
  );
}
