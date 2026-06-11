import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Link } from 'react-router-dom';
import OpenSeadragon from 'openseadragon';
import { createRoot, type Root } from 'react-dom/client';

import OpenSeadragonViewer, {
  type TileSourceConfig,
} from '../components/OpenSeadragonViewer';
import MapMarker, { type MarkerTag } from '../components/MapMarker';
import {
  setupDraggableOverlay,
  type ImageDimensions,
} from '../hooks/useOpenSeadragonDrag';
import { useTenant } from '../contexts/TenantContext';
import client from '../api/client';
import type { components } from '../api/schema';

import styles from './PointPickerPage.module.css';

// ------------------------------------------------------------------ types

type MapSummary = components['schemas']['MapSummaryOut'];

interface PointData {
  id: string;
  tag: MarkerTag;
  pixel_x: number;
  pixel_y: number;
}

interface ImageInfo {
  filename: string;
  width: number;
  height: number;
  geotransform: number[] | null;
}

interface OverlayEntry {
  point: PointData;
  element: HTMLDivElement;
  root: Root;
  tracker: OpenSeadragon.MouseTracker;
}

const TAG_LABELS: Record<MarkerTag, string> = {
  parking_spot: 'Parking Spot',
  arrows: 'Arrows',
  crosswalk: 'Crosswalk',
  extra: 'Extra',
};

const TAGS: MarkerTag[] = ['parking_spot', 'arrows', 'crosswalk', 'extra'];

const TAG_ACTIVE_STYLE: Record<MarkerTag, string> = {
  parking_spot: styles.tagBtnParkingSpot,
  arrows: styles.tagBtnArrows,
  crosswalk: styles.tagBtnCrosswalk,
  extra: styles.tagBtnExtra,
};

// ----------------------------------------------------------- component

export default function PointPickerPage() {
  const { selectedTenantId } = useTenant();

  const [maps, setMaps] = useState<MapSummary[]>([]);
  const [selectedMapId, setSelectedMapId] = useState<string | null>(null);
  const [imageInfo, setImageInfo] = useState<ImageInfo | null>(null);
  const [currentTag, setCurrentTag] = useState<MarkerTag>('parking_spot');
  const [nextId, setNextId] = useState<string>('');
  const [coordsText, setCoordsText] = useState('Hover over image...');
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);
  const [points, setPoints] = useState<PointData[]>([]);

  const viewerRef = useRef<OpenSeadragon.Viewer | null>(null);
  const overlaysRef = useRef<Map<string, OverlayEntry>>(new Map());
  const isDraggingRef = useRef(false);
  const coordsTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep selectedPointId in a ref so overlay callbacks always see the latest value.
  const selectedPointIdRef = useRef(selectedPointId);
  useEffect(() => {
    selectedPointIdRef.current = selectedPointId;
  }, [selectedPointId]);

  // Keep selectedMapId in a ref so overlay drag callbacks (memoised without
  // selectedMapId in their deps) always persist against the current map.
  const selectedMapIdRef = useRef(selectedMapId);
  useEffect(() => {
    selectedMapIdRef.current = selectedMapId;
  }, [selectedMapId]);

  // --------------------------------------------------- selected map state
  const selectedMap = useMemo(
    () => maps.find((m) => m.id === selectedMapId) ?? null,
    [maps, selectedMapId],
  );
  const imageConfigured = selectedMap?.image_configured ?? false;

  // ----------------------------------------------------------- load maps
  // Fetch the maps available for the current tenant and pick a sensible
  // default (the first image-configured map, else the first map).
  useEffect(() => {
    const controller = new AbortController();

    (async () => {
      if (!selectedTenantId) {
        setMaps([]);
        setSelectedMapId(null);
        return;
      }
      const { data } = await client.GET('/point-picker/api/maps/', {
        params: { query: { tenant_id: '' } },
        signal: controller.signal,
      });
      if (controller.signal.aborted || !data) return;
      const list = data.maps;
      setMaps(list);
      const preferred = list.find((m) => m.image_configured) ?? list[0] ?? null;
      setSelectedMapId(preferred ? preferred.id : null);
    })().catch((err: unknown) => {
      if (!controller.signal.aborted) {
        console.error('Failed to load maps:', err);
      }
    });

    return () => { controller.abort(); };
  }, [selectedTenantId]);

  // ------------------------------------------------- fetch helpers (raw)
  // The openapi-fetch client auto-injects tenant_id and auth. We use it
  // for typed JSON endpoints. Tile URLs are bare strings for OSD.

  const fetchNextId = useCallback(
    async (tag: MarkerTag) => {
      const { data } = await client.GET('/point-picker/api/points/next-id/', {
        params: { query: { tag, tenant_id: '', map_id: selectedMapId } },
      });
      if (data) setNextId(data.next_id);
    },
    [selectedMapId],
  );

  // ----------------------------------------------------- load image info
  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (!selectedTenantId || !selectedMapId || !imageConfigured) {
        setImageInfo(null);
        return;
      }
      const { data } = await client.GET('/point-picker/api/image/info/', {
        params: { query: { tenant_id: '', map_id: selectedMapId } },
      });
      if (cancelled || !data) return;
      setImageInfo({
        filename: data.filename,
        width: data.width,
        height: data.height,
        geotransform: data.geotransform ?? null,
      });
    })();

    return () => { cancelled = true; };
  }, [selectedTenantId, selectedMapId, imageConfigured]);

  // --------------------------------------------------------- tile source
  // Memoise so we only hand OSD a new object when tenant or image actually
  // change. The tile URL must embed tenant_id directly because OSD fetches
  // tiles via <img src>, not through the openapi-fetch client.
  const tileSource: TileSourceConfig | null = useMemo(() => {
    if (!imageInfo || !selectedTenantId || !selectedMapId || !imageConfigured) {
      return null;
    }
    return {
      width: imageInfo.width,
      height: imageInfo.height,
      tileSize: 256,
      getTileUrl: (level: number, x: number, y: number) => {
        const params = new URLSearchParams({
          x: String(x),
          y: String(y),
          z: String(level),
          tenant_id: selectedTenantId,
          map_id: selectedMapId,
        });
        return `/point-picker/api/image/tile/?${params}`;
      },
    };
  }, [imageInfo, selectedTenantId, selectedMapId, imageConfigured]);

  // ------------------------------------------------- image dimensions ref
  const imageDims: ImageDimensions | null = useMemo(() => {
    if (!imageInfo) return null;
    return { width: imageInfo.width, height: imageInfo.height };
  }, [imageInfo]);

  // -------------------------------------------------------- load points
  const loadPoints = useCallback(async () => {
    const requestedMapId = selectedMapId;
    const { data } = await client.GET('/point-picker/api/points/', {
      params: { query: { tenant_id: '', map_id: requestedMapId } },
    });
    // Drop a stale response if the selected map changed mid-flight, so a slow
    // load for map A cannot overwrite the points already shown for map B.
    if (requestedMapId !== selectedMapIdRef.current) return;
    if (data) {
      setPoints(data.points as PointData[]);
    }
  }, [selectedMapId]);

  // -------------------------------------------------------- tag selection
  const handleSelectTag = useCallback(
    (tag: MarkerTag) => {
      setCurrentTag(tag);
      fetchNextId(tag);
    },
    [fetchNextId],
  );

  // Initial tag fetch
  useEffect(() => {
    if (!selectedTenantId || !selectedMapId || !imageConfigured) return;
    void (async () => { await fetchNextId(currentTag); })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTenantId, selectedMapId, imageConfigured]);

  // ------------------------------------------------ marker management

  /** Render a MapMarker into a detached DOM element for use as an OSD overlay. */
  const renderMarkerElement = useCallback(
    (point: PointData, isSelected: boolean): { element: HTMLDivElement; root: Root } => {
      const element = document.createElement('div');
      const root = createRoot(element);
      root.render(
        <MapMarker
          id={point.id}
          tag={point.tag}
          selected={isSelected}
        />,
      );
      return { element, root };
    },
    [],
  );

  /** Update the React tree inside a marker overlay to toggle selection. */
  const updateMarkerSelection = useCallback(
    (overlayEntry: OverlayEntry, isSelected: boolean) => {
      overlayEntry.root.render(
        <MapMarker
          id={overlayEntry.point.id}
          tag={overlayEntry.point.tag}
          selected={isSelected}
        />,
      );
    },
    [],
  );

  /** Select a point: update overlay visuals and component state. */
  const selectPoint = useCallback(
    (pointId: string | null) => {
      const prev = selectedPointIdRef.current;
      const overlays = overlaysRef.current;

      // Deselect previous
      if (prev && overlays.has(prev)) {
        updateMarkerSelection(overlays.get(prev)!, false);
      }
      // Select new
      if (pointId && overlays.has(pointId)) {
        updateMarkerSelection(overlays.get(pointId)!, true);
      }
      setSelectedPointId(pointId);
    },
    [updateMarkerSelection],
  );

  /** Add a single overlay marker to the viewer. */
  const addOverlayMarker = useCallback(
    (point: PointData) => {
      const viewer = viewerRef.current;
      if (!viewer || !imageDims) return;

      const isSelected = selectedPointIdRef.current === point.id;
      const { element, root } = renderMarkerElement(point, isSelected);

      const location = viewer.viewport.imageToViewportCoordinates(
        point.pixel_x,
        point.pixel_y,
      );
      viewer.addOverlay({
        element,
        location,
        placement: OpenSeadragon.Placement.BOTTOM,
      });

      const tracker = setupDraggableOverlay(viewer, element, imageDims, {
        onDragStart: () => {
          isDraggingRef.current = true;
          selectPoint(point.id);
        },
        onDrag: (x, y) => {
          const loc = viewer.viewport.imageToViewportCoordinates(x, y);
          viewer.updateOverlay(element, loc);
        },
        onDragEnd: async (x, y) => {
          isDraggingRef.current = false;
          // Persist move to server
          await client.PUT('/point-picker/api/points/{point_id}/', {
            params: {
              query: { tenant_id: '', map_id: selectedMapIdRef.current },
              path: { point_id: point.id },
            },
            body: { pixel_x: x, pixel_y: y },
          });
          // Update local state
          setPoints((prev) =>
            prev.map((p) =>
              p.id === point.id ? { ...p, pixel_x: x, pixel_y: y } : p,
            ),
          );
        },
      });

      // Click on marker selects it (without adding a new point)
      const clickTracker = new OpenSeadragon.MouseTracker({
        element,
        clickHandler: () => {
          selectPoint(point.id);
        },
      });

      overlaysRef.current.set(point.id, {
        point,
        element,
        root,
        tracker: Object.assign(tracker, { _clickTracker: clickTracker }),
      });
    },
    [imageDims, renderMarkerElement, selectPoint],
  );

  /** Remove a single overlay marker from the viewer. */
  const removeOverlayMarker = useCallback((pointId: string) => {
    const viewer = viewerRef.current;
    const entry = overlaysRef.current.get(pointId);
    if (!entry) return;

    entry.tracker.destroy();
    // Also destroy the click tracker we attached
    const ct = (entry.tracker as unknown as { _clickTracker?: OpenSeadragon.MouseTracker })
      ._clickTracker;
    if (ct) ct.destroy();

    if (viewer) viewer.removeOverlay(entry.element);
    entry.root.unmount();
    overlaysRef.current.delete(pointId);
  }, []);

  // Sync overlays when points array changes
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !imageDims) return;

    const currentIds = new Set(points.map((p) => p.id));
    const overlayIds = new Set(overlaysRef.current.keys());

    // Remove stale overlays
    for (const id of overlayIds) {
      if (!currentIds.has(id)) {
        removeOverlayMarker(id);
      }
    }

    // Add missing overlays
    for (const point of points) {
      if (!overlayIds.has(point.id)) {
        addOverlayMarker(point);
      } else {
        // Update position if changed
        const entry = overlaysRef.current.get(point.id)!;
        if (
          entry.point.pixel_x !== point.pixel_x ||
          entry.point.pixel_y !== point.pixel_y
        ) {
          const loc = viewer.viewport.imageToViewportCoordinates(
            point.pixel_x,
            point.pixel_y,
          );
          viewer.updateOverlay(entry.element, loc);
          entry.point = point;
        }
      }
    }
  }, [points, imageDims, addOverlayMarker, removeOverlayMarker]);

  // ------------------------------------ load points when tenant/map ready
  useEffect(() => {
    void (async () => {
      if (selectedTenantId && selectedMapId && imageConfigured) {
        await loadPoints();
      } else {
        setPoints([]);
      }
    })();
  }, [selectedTenantId, selectedMapId, imageConfigured, loadPoints]);

  // --------------------------------------------------- add point on click
  const handleCanvasClick = useCallback(
    async (imageX: number, imageY: number) => {
      if (isDraggingRef.current) return;
      if (!imageInfo || !imageConfigured) return;
      if (imageX < 0 || imageX >= imageInfo.width || imageY < 0 || imageY >= imageInfo.height) {
        return;
      }

      const { data } = await client.POST('/point-picker/api/points/', {
        params: { query: { tenant_id: '', map_id: selectedMapId } },
        body: { tag: currentTag, pixel_x: imageX, pixel_y: imageY },
      });
      if (data) {
        const newPoint: PointData = data as PointData;
        setPoints((prev) => [...prev, newPoint]);
        fetchNextId(currentTag);
      }
    },
    [imageInfo, imageConfigured, currentTag, fetchNextId, selectedMapId],
  );

  // -------------------------------------------- coordinate display on move
  const handleMouseMove = useCallback(
    (imageX: number, imageY: number) => {
      if (!imageInfo) return;

      if (
        imageX < 0 ||
        imageX >= imageInfo.width ||
        imageY < 0 ||
        imageY >= imageInfo.height
      ) {
        setCoordsText('Outside image');
        return;
      }

      let text = `Pixel: ${imageX.toFixed(1)}, ${imageY.toFixed(1)}`;
      setCoordsText(text);

      // Debounced geo-coords fetch
      if (coordsTimerRef.current) clearTimeout(coordsTimerRef.current);
      if (imageInfo.geotransform) {
        coordsTimerRef.current = setTimeout(async () => {
          const { data } = await client.GET('/point-picker/api/geo-coords/', {
            params: {
              query: {
                tenant_id: '',
                map_id: selectedMapIdRef.current,
                pixel_x: imageX,
                pixel_y: imageY,
              },
            },
          });
          if (data && data.easting != null && data.northing != null) {
            text += `\nGeo: ${data.easting.toFixed(2)}, ${data.northing.toFixed(2)}`;
            if (data.crs) text += `\nCRS: ${data.crs}`;
            setCoordsText(text);
          }
        }, 50);
      }
    },
    [imageInfo],
  );

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (coordsTimerRef.current) clearTimeout(coordsTimerRef.current);
    };
  }, []);

  // ---------------------------------------------------------- delete point
  const handleDeletePoint = useCallback(
    async (pointId: string) => {
      const { data } = await client.DELETE(
        '/point-picker/api/points/{point_id}/',
        {
          params: {
            query: { tenant_id: '', map_id: selectedMapId },
            path: { point_id: pointId },
          },
        },
      );
      if (data) {
        setPoints((prev) => prev.filter((p) => p.id !== pointId));
        if (selectedPointId === pointId) setSelectedPointId(null);
        fetchNextId(currentTag);
      }
    },
    [selectedPointId, fetchNextId, currentTag, selectedMapId],
  );

  // ---------------------------------------------------- click point in list
  const handlePointListClick = useCallback(
    (point: PointData) => {
      selectPoint(point.id);
      const viewer = viewerRef.current;
      if (!viewer) return;
      const location = viewer.viewport.imageToViewportCoordinates(
        point.pixel_x,
        point.pixel_y,
      );
      viewer.viewport.panTo(location, false);
      if (viewer.viewport.getZoom() < 2) {
        viewer.viewport.zoomTo(2, location, false);
      }
    },
    [selectPoint],
  );

  // --------------------------------------------------------- viewer ready
  const handleViewerReady = useCallback(
    (viewer: OpenSeadragon.Viewer) => {
      viewerRef.current = viewer;
    },
    [],
  );

  // Cleanup overlays on unmount
  useEffect(() => {
    return () => {
      for (const id of overlaysRef.current.keys()) {
        removeOverlayMarker(id);
      }
    };
  }, [removeOverlayMarker]);

  // --------------------------------------------------------------- render

  if (!selectedTenantId) {
    return (
      <div className={styles.page}>
        <p style={{ padding: 16 }}>Select a tenant to continue.</p>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      {/* ----- Sidebar ----- */}
      <div className={styles.sidebar}>
        <Link to="/" className={styles.backLink}>
          &larr; Back to Tools
        </Link>
        <h1 className={styles.title}>Map Points Capture Tool</h1>

        {/* Site / map selector */}
        {maps.length > 0 && (
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Site</div>
            <select
              className={styles.mapSelect}
              value={selectedMapId ?? ''}
              onChange={(e) => setSelectedMapId(e.target.value)}
            >
              {maps.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                  {m.image_configured ? '' : ' (no image)'}
                </option>
              ))}
            </select>
          </div>
        )}

        {imageConfigured && imageInfo && (
          <div className={styles.filename}>{imageInfo.filename}</div>
        )}

        {/* Tag category */}
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Tag Category</div>
          <div className={styles.tagButtons}>
            {TAGS.map((tag) => {
              const isActive = currentTag === tag;
              const cls = [
                styles.tagBtn,
                isActive ? TAG_ACTIVE_STYLE[tag] : '',
              ]
                .filter(Boolean)
                .join(' ');
              return (
                <button
                  key={tag}
                  type="button"
                  className={cls}
                  disabled={!imageConfigured}
                  onClick={() => handleSelectTag(tag)}
                >
                  {TAG_LABELS[tag]}
                </button>
              );
            })}
          </div>
          <div className={styles.nextId}>Next: {nextId}</div>
        </div>

        {/* Coordinates */}
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Coordinates</div>
          <div className={styles.coords}>{coordsText}</div>
        </div>

        {/* Point list */}
        {imageConfigured && (
          <div className={styles.section}>
            <div className={styles.sectionTitle}>Points</div>
            <div className={styles.pointList}>
              {points.map((pt) => {
                const isSelected = pt.id === selectedPointId;
                const cls = [
                  styles.pointItem,
                  isSelected ? styles.pointItemSelected : '',
                ]
                  .filter(Boolean)
                  .join(' ');
                return (
                  <div
                    key={pt.id}
                    className={cls}
                    onClick={() => handlePointListClick(pt)}
                  >
                    <span>
                      {pt.id}: ({pt.pixel_x.toFixed(1)}, {pt.pixel_y.toFixed(1)})
                    </span>
                    <button
                      type="button"
                      className={styles.deleteBtn}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeletePoint(pt.id);
                      }}
                    >
                      X
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ----- Viewer / unconfigured panel ----- */}
      {imageConfigured ? (
        tileSource && (
          <OpenSeadragonViewer
            tileSource={tileSource}
            onViewerReady={handleViewerReady}
            onCanvasClick={handleCanvasClick}
            onMouseMove={handleMouseMove}
            className={styles.viewer}
            showNavigator
            minZoomLevel={0.5}
            maxZoomLevel={20}
          />
        )
      ) : (
        <div className={styles.viewer}>
          <div className={styles.notConfigured}>
            <div className={styles.notConfiguredTitle}>
              &#9888; Image not configured for this site
            </div>
            <p className={styles.notConfiguredBody}>
              The site &ldquo;{selectedMap?.label ?? ''}&rdquo; does not have a
              reference image configured. Map point capture is disabled until an
              image is added.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
