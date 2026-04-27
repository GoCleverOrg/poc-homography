/**
 * OpenSeadragon overlay drag utilities (TypeScript port).
 *
 * These are plain imperative functions -- not React hooks -- because they
 * operate directly on OSD's MouseTracker API and are created / destroyed
 * by the parent component in response to viewer lifecycle events.
 *
 * Ported from `webapp/static/js/openseadragon-drag-utils.js`.
 *
 * Why we avoid `e.position` during drag:
 *   With `dragOutside: true`, OSD's `e.position` becomes unreliable once the
 *   pointer leaves the overlay element bounds (coordinates are relative to the
 *   overlay, not the viewer).  Using `e.originalEvent.clientX/clientY` and
 *   manually converting through the viewer's bounding rect fixes this.
 */
import OpenSeadragon from 'openseadragon';

// ------------------------------------------------------------------ types

export interface ImageCoords {
  x: number;
  y: number;
  inBounds: boolean;
}

export interface ImageDimensions {
  width: number;
  height: number;
}

export interface DraggableOverlayCallbacks {
  onDragStart?: (element: HTMLElement) => void;
  onDrag?: (x: number, y: number, element: HTMLElement) => void;
  onDragEnd?: (x: number, y: number, element: HTMLElement) => void;
}

/**
 * The MouseTracker event types use `originalEvent: Event` in the type defs,
 * but at runtime it is always the browser's native pointer/mouse event.
 * We narrow the type here so callers get proper `clientX`/`clientY`.
 */
interface DragLikeEvent {
  originalEvent?: Event;
}

// --------------------------------------------------------------- helpers

/**
 * Convert a drag / release event's pointer position to image-pixel
 * coordinates.
 *
 * Always uses `e.originalEvent.clientX/clientY` rather than `e.position`
 * to work around the `dragOutside` coordinate bug (see module docstring).
 *
 * @param clamp  If `true` (default), the returned coordinates are clamped
 *               to `[0 .. width-1]` / `[0 .. height-1]`.
 */
export function getImageCoordsFromDragEvent(
  viewer: OpenSeadragon.Viewer,
  e: DragLikeEvent,
  imageInfo: ImageDimensions,
  clamp = true,
): ImageCoords | null {
  const mouseEvent = e.originalEvent as MouseEvent | undefined;
  if (!mouseEvent || typeof mouseEvent.clientX !== 'number') {
    return null;
  }

  // Pointer position relative to the viewer DOM element
  const viewerRect = viewer.element.getBoundingClientRect();
  const viewerX = mouseEvent.clientX - viewerRect.left;
  const viewerY = mouseEvent.clientY - viewerRect.top;

  // Viewer pixels -> viewport coords -> image coords
  const viewportPoint = viewer.viewport.pointFromPixel(
    new OpenSeadragon.Point(viewerX, viewerY),
  );
  const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);

  let x = imagePoint.x;
  let y = imagePoint.y;

  const inBounds =
    x >= 0 && x < imageInfo.width && y >= 0 && y < imageInfo.height;

  if (clamp) {
    x = Math.max(0, Math.min(imageInfo.width - 1, x));
    y = Math.max(0, Math.min(imageInfo.height - 1, y));
  }

  return { x, y, inBounds };
}

/**
 * Attach draggable behaviour to an OSD overlay element.
 *
 * Returns the `MouseTracker` instance so the caller can `.destroy()` it
 * when the overlay is removed.
 */
export function setupDraggableOverlay(
  viewer: OpenSeadragon.Viewer,
  element: HTMLElement,
  imageInfo: ImageDimensions,
  callbacks: DraggableOverlayCallbacks = {},
): OpenSeadragon.MouseTracker {
  const { onDragStart, onDrag, onDragEnd } = callbacks;
  let isDragging = false;

  const tracker = new OpenSeadragon.MouseTracker({
    element,

    pressHandler: () => {
      isDragging = true;
      onDragStart?.(element);
    },

    dragHandler: (e) => {
      if (!isDragging) return;
      const coords = getImageCoordsFromDragEvent(
        viewer,
        e as unknown as DragLikeEvent,
        imageInfo,
      );
      if (coords) onDrag?.(coords.x, coords.y, element);
    },

    releaseHandler: (e) => {
      if (!isDragging) return;
      isDragging = false;
      const coords = getImageCoordsFromDragEvent(
        viewer,
        e as unknown as DragLikeEvent,
        imageInfo,
      );
      if (coords) onDragEnd?.(coords.x, coords.y, element);
    },

    // CRITICAL: allows drag to continue when pointer leaves the small element
    // @ts-expect-error -- dragOutside exists at runtime but is missing from @types/openseadragon
    dragOutside: true,
  });

  return tracker;
}
