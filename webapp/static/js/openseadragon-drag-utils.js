/**
 * OpenSeadragon Overlay Drag Utilities
 *
 * PROBLEM SOLVED:
 * When using OpenSeadragon's MouseTracker with `dragOutside: true` to drag overlay
 * elements (markers, points, etc.), the `e.position` property becomes UNRELIABLE
 * once the mouse leaves the overlay element's bounds. It returns coordinates
 * relative to the overlay element's origin, not the viewer - causing the marker
 * to flicker between the cursor position and the top-left corner.
 *
 * SOLUTION:
 * Always use `e.originalEvent.clientX/clientY` and manually calculate the position
 * relative to the viewer element. This works correctly at any zoom/pan level.
 *
 * USAGE EXAMPLE:
 * ```javascript
 * new OpenSeadragon.MouseTracker({
 *     element: markerElement,
 *     pressHandler: (e) => { ... },
 *     dragHandler: (e) => {
 *         const coords = getImageCoordsFromDragEvent(viewer, e, imageInfo);
 *         if (coords) {
 *             viewer.updateOverlay(marker, viewer.viewport.imageToViewportCoordinates(coords.x, coords.y));
 *         }
 *     },
 *     releaseHandler: (e) => { ... },
 *     dragOutside: true  // CRITICAL: enables dragging outside the small element
 * });
 * ```
 *
 * @module openseadragon-drag-utils
 */

/**
 * Get image coordinates from an OpenSeadragon drag/release event.
 *
 * DO NOT use `e.position` directly with `dragOutside: true` - it will give
 * incorrect coordinates when the mouse is outside the overlay element.
 *
 * @param {OpenSeadragon.Viewer} viewer - The OpenSeadragon viewer instance
 * @param {Object} e - The OpenSeadragon MouseTracker event object
 * @param {Object} imageInfo - Object containing image dimensions {width, height}
 * @param {boolean} [clamp=true] - Whether to clamp coordinates to image bounds
 * @returns {{x: number, y: number, inBounds: boolean}|null} Image coordinates or null if no originalEvent
 */
function getImageCoordsFromDragEvent(viewer, e, imageInfo, clamp = true) {
    // Must use originalEvent - e.position is unreliable with dragOutside
    const mouseEvent = e.originalEvent;
    if (!mouseEvent) {
        console.warn('getImageCoordsFromDragEvent: No originalEvent found');
        return null;
    }

    // Get mouse position relative to viewer element
    const viewerRect = viewer.element.getBoundingClientRect();
    const viewerX = mouseEvent.clientX - viewerRect.left;
    const viewerY = mouseEvent.clientY - viewerRect.top;

    // Convert viewer pixels → viewport coordinates → image coordinates
    const viewportPoint = viewer.viewport.pointFromPixel(
        new OpenSeadragon.Point(viewerX, viewerY)
    );
    const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint);

    let x = imagePoint.x;
    let y = imagePoint.y;

    // Check if within bounds before clamping
    const inBounds = x >= 0 && x < imageInfo.width && y >= 0 && y < imageInfo.height;

    // Clamp to image bounds if requested
    if (clamp) {
        x = Math.max(0, Math.min(imageInfo.width - 1, x));
        y = Math.max(0, Math.min(imageInfo.height - 1, y));
    }

    return { x, y, inBounds };
}

/**
 * Set up draggable behavior for an OpenSeadragon overlay element.
 *
 * This is a convenience wrapper that handles all the coordinate conversion
 * correctly. Provide callbacks for drag start, drag move, and drag end.
 *
 * @param {OpenSeadragon.Viewer} viewer - The OpenSeadragon viewer instance
 * @param {HTMLElement} element - The overlay element to make draggable
 * @param {Object} imageInfo - Object containing image dimensions {width, height}
 * @param {Object} callbacks - Callback functions
 * @param {Function} [callbacks.onDragStart] - Called when drag starts, receives (element)
 * @param {Function} [callbacks.onDrag] - Called during drag, receives (x, y, element)
 * @param {Function} [callbacks.onDragEnd] - Called when drag ends, receives (x, y, element)
 * @returns {OpenSeadragon.MouseTracker} The MouseTracker instance (for cleanup)
 *
 * @example
 * const tracker = setupDraggableOverlay(viewer, markerEl, imageInfo, {
 *     onDragStart: (el) => el.classList.add('dragging'),
 *     onDrag: (x, y, el) => {
 *         viewer.updateOverlay(el, viewer.viewport.imageToViewportCoordinates(x, y));
 *         updateVisuals();
 *     },
 *     onDragEnd: async (x, y, el) => {
 *         el.classList.remove('dragging');
 *         await savePosition(x, y);
 *     }
 * });
 */
function setupDraggableOverlay(viewer, element, imageInfo, callbacks = {}) {
    const { onDragStart, onDrag, onDragEnd } = callbacks;
    let isDragging = false;

    const tracker = new OpenSeadragon.MouseTracker({
        element: element,

        pressHandler: (e) => {
            isDragging = true;
            if (onDragStart) {
                onDragStart(element);
            }
        },

        dragHandler: (e) => {
            if (!isDragging) return;

            const coords = getImageCoordsFromDragEvent(viewer, e, imageInfo);
            if (coords && onDrag) {
                onDrag(coords.x, coords.y, element);
            }
        },

        releaseHandler: async (e) => {
            if (!isDragging) return;
            isDragging = false;

            const coords = getImageCoordsFromDragEvent(viewer, e, imageInfo);
            if (coords && onDragEnd) {
                await onDragEnd(coords.x, coords.y, element);
            }
        },

        // CRITICAL: This allows dragging to continue when mouse leaves the small element
        dragOutside: true
    });

    return tracker;
}

// Export for module systems (if used)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { getImageCoordsFromDragEvent, setupDraggableOverlay };
}
