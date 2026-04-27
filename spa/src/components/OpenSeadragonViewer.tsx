import { useRef, useEffect, useCallback } from 'react';
import OpenSeadragon from 'openseadragon';

export interface TileSourceConfig {
  width: number;
  height: number;
  tileSize?: number;
  getTileUrl: (level: number, x: number, y: number) => string;
}

export interface OpenSeadragonViewerProps {
  tileSource: TileSourceConfig;
  onViewerReady?: (viewer: OpenSeadragon.Viewer) => void;
  onCanvasClick?: (imageX: number, imageY: number) => void;
  onMouseMove?: (imageX: number, imageY: number) => void;
  className?: string;
  showNavigator?: boolean;
  minZoomLevel?: number;
  maxZoomLevel?: number;
}

/**
 * OSD navigation button images shipped with the npm package.
 * Using a CDN fallback keeps things working even when the local
 * path is not served by Vite's dev server.
 */
const PREFIX_URL =
  'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/';

/**
 * Build the tile-source config object that OSD's constructor and `.open()`
 * both understand at runtime.  The v6 type definitions split this into
 * several incompatible union branches, so we cast to `object` at the
 * boundary.
 */
function buildTileSource(cfg: TileSourceConfig): object {
  return {
    height: cfg.height,
    width: cfg.width,
    tileSize: cfg.tileSize ?? 256,
    getTileUrl: cfg.getTileUrl,
  };
}

export default function OpenSeadragonViewer({
  tileSource,
  onViewerReady,
  onCanvasClick,
  onMouseMove,
  className,
  showNavigator = true,
  minZoomLevel = 0.5,
  maxZoomLevel = 20,
}: OpenSeadragonViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);

  // Keep callback refs stable so event handlers always call the latest prop.
  // Assignments live in useEffect to satisfy react-hooks/refs (React 19).
  const onCanvasClickRef = useRef(onCanvasClick);
  const onMouseMoveRef = useRef(onMouseMove);
  const onViewerReadyRef = useRef(onViewerReady);

  useEffect(() => {
    onCanvasClickRef.current = onCanvasClick;
    onMouseMoveRef.current = onMouseMove;
    onViewerReadyRef.current = onViewerReady;
  });

  // Stable helper: convert a canvas-level pixel position to image coordinates.
  const toImageCoords = useCallback(
    (viewer: OpenSeadragon.Viewer, position: OpenSeadragon.Point) => {
      const viewportPoint = viewer.viewport.pointFromPixel(position);
      return viewer.viewport.viewportToImageCoordinates(viewportPoint);
    },
    [],
  );

  // ---------- create viewer once on mount ----------
  useEffect(() => {
    if (!containerRef.current) return;

    const viewer = OpenSeadragon({
      element: containerRef.current,
      prefixUrl: PREFIX_URL,
      tileSources: buildTileSource(tileSource) as OpenSeadragon.TileSourceOptions,
      showNavigator,
      navigatorPosition: 'BOTTOM_RIGHT',
      minZoomLevel,
      maxZoomLevel,
      visibilityRatio: 0.5,
      constrainDuringPan: true,
    });

    viewerRef.current = viewer;

    // --- canvas-click -> image coords ---
    viewer.addHandler('canvas-click', (event) => {
      if (!onCanvasClickRef.current) return;
      const imagePoint = toImageCoords(viewer, event.position);
      onCanvasClickRef.current(imagePoint.x, imagePoint.y);
    });

    // --- mouse move (throttled via rAF) -> image coords ---
    // There is no "canvas-move" viewer event, so we attach a separate
    // MouseTracker to the viewer's canvas element.
    let rafId = 0;
    const moveTracker = new OpenSeadragon.MouseTracker({
      element: viewer.canvas,
      moveHandler: (event) => {
        if (!onMouseMoveRef.current) return;
        if (rafId) return; // already scheduled

        rafId = requestAnimationFrame(() => {
          rafId = 0;
          if (!onMouseMoveRef.current) return;
          // event.position is reliable here (no dragOutside concern)
          const mouseEvent = event.originalEvent as MouseEvent | undefined;
          if (!mouseEvent || typeof mouseEvent.clientX !== 'number') return;

          const rect = viewer.element.getBoundingClientRect();
          const vx = mouseEvent.clientX - rect.left;
          const vy = mouseEvent.clientY - rect.top;
          const viewportPt = viewer.viewport.pointFromPixel(
            new OpenSeadragon.Point(vx, vy),
          );
          const imgPt =
            viewer.viewport.viewportToImageCoordinates(viewportPt);
          onMouseMoveRef.current(imgPt.x, imgPt.y);
        });
      },
    });

    onViewerReadyRef.current?.(viewer);

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      moveTracker.destroy();
      viewer.destroy();
      viewerRef.current = null;
    };
    // Only run on mount / unmount. Tile source swaps are handled below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- swap tile source when the config identity changes ----------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    // OSD v6 `open()` expects a TileSourceSpecifier with a `tileSource` key.
    const spec: OpenSeadragon.TileSourceSpecifier = {
      tileSource: buildTileSource(tileSource),
    };
    viewer.open(spec);
  }, [tileSource]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{ width: '100%', height: '100%' }}
    />
  );
}
