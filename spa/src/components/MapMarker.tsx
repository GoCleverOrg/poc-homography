import { forwardRef } from 'react';
import styles from './MapMarker.module.css';

export type MarkerTag = 'parking_spot' | 'arrows' | 'crosswalk' | 'extra';

export interface MapMarkerProps {
  /** Unique point identifier displayed inside the pin, e.g. "P3". */
  id: string;
  /** Ground-control-point tag type -- determines the pin colour. */
  tag: MarkerTag;
  /** Whether this marker is currently selected (yellow glow). */
  selected?: boolean;
  /** Short text rendered inside the pin head. Falls back to numeric suffix of `id`. */
  label?: string;
}

/** Map from domain tag names to CSS-module class names. */
const TAG_CLASS: Record<MarkerTag, string> = {
  parking_spot: styles.parkingSpot,
  arrows: styles.arrows,
  crosswalk: styles.crosswalk,
  extra: styles.extra,
};

/**
 * CSS-only pin marker for OSD overlays.
 *
 * The component renders a small coloured pin with an optional label.
 * It is designed to be used with `viewer.addOverlay()` -- the parent
 * obtains the DOM node via `ref` and hands it to OpenSeadragon.
 *
 * ```tsx
 * const markerRef = useRef<HTMLDivElement>(null);
 * // after viewer is ready:
 * viewer.addOverlay({
 *   element: markerRef.current!,
 *   location: viewer.viewport.imageToViewportCoordinates(px, py),
 *   placement: OpenSeadragon.Placement.BOTTOM,
 * });
 * ```
 */
const MapMarker = forwardRef<HTMLDivElement, MapMarkerProps>(
  function MapMarker({ id, tag, selected = false, label }, ref) {
    const tagClass = TAG_CLASS[tag] ?? TAG_CLASS.extra;
    const displayLabel = label ?? id.replace(/[A-Za-z]+/, '');

    const className = [
      styles.marker,
      tagClass,
      selected ? styles.selected : '',
    ]
      .filter(Boolean)
      .join(' ');

    return (
      <div ref={ref} className={className} data-point-id={id}>
        <span className={styles.label}>{displayLabel}</span>
      </div>
    );
  },
);

export default MapMarker;
