#!/usr/bin/env python3
"""
Interactive tool to extract reference points from a georeferenced image and export to KML.
Click on features (zebra crossings, arrows, parking corners) and save as KML.
"""

import argparse
import base64
import http.server
import json
import socketserver
import sys
import webbrowser
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from poc_homography.camera_config import get_camera_by_name, get_camera_configs
from poc_homography.kml import GeoConfig, Kml, PointExtractor
from poc_homography.server_utils import find_available_port


def create_html(image_path: str, config: dict) -> str:
    """Create the HTML interface."""

    # Read and encode image
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    # Determine image type
    suffix = Path(image_path).suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "image/png")

    return (
        """<!DOCTYPE html>
<html>
<head>
    <title>KML Point Extractor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; }
        .container { display: flex; height: 100vh; }
        .image-panel { flex: 1; overflow: auto; position: relative; background: #16213e; }
        .sidebar { width: 350px; background: #0f3460; padding: 15px; overflow-y: auto; }

        #image-container {
            position: relative;
            display: inline-block;
            cursor: crosshair;
        }
        #main-image { display: block; max-width: none; }

        .marker {
            position: absolute;
            padding: 2px 5px;
            border-radius: 3px;
            border: 2px solid white;
            cursor: pointer;
            font-size: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            white-space: nowrap;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.8);
            z-index: 10;
        }
        .marker.zebra { background: #e94560; }
        .marker.arrow { background: #0ead69; }
        .marker.parking { background: #3498db; }
        .marker.other { background: #f39c12; }

        .point-dot {
            position: absolute;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            border: 2px solid white;
            transform: translate(-50%, -50%);
            z-index: 5;
        }
        .point-dot.zebra { background: #e94560; }
        .point-dot.arrow { background: #0ead69; }
        .point-dot.parking { background: #3498db; }
        .point-dot.other { background: #f39c12; }

        #connectors-svg {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 4;
        }
        .connector-path {
            fill: none;
            stroke: rgba(255,255,255,0.8);
            stroke-width: 2;
        }

        h2 { margin-bottom: 15px; color: #e94560; }
        .controls { margin-bottom: 20px; }
        label { display: block; margin: 10px 0 5px; font-size: 12px; color: #aaa; }
        select, input, button {
            width: 100%; padding: 8px; margin-bottom: 10px;
            border: 1px solid #333; border-radius: 4px;
            background: #16213e; color: #eee;
        }
        button {
            background: #e94560; border: none; cursor: pointer;
            font-weight: bold; transition: background 0.2s;
        }
        button:hover { background: #c73e54; }
        button.secondary { background: #0ead69; }
        button.secondary:hover { background: #0c9a5c; }

        .point-list { margin-top: 20px; }

        .category-filters {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .category-btn {
            padding: 5px 10px;
            border: 2px solid;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-weight: bold;
            transition: opacity 0.2s;
        }
        .category-btn.zebra { background: #e94560; border-color: #e94560; color: white; }
        .category-btn.arrow { background: #0ead69; border-color: #0ead69; color: white; }
        .category-btn.parking { background: #3498db; border-color: #3498db; color: white; }
        .category-btn.other { background: #f39c12; border-color: #f39c12; color: white; }
        .category-btn.hidden {
            background: transparent;
            opacity: 0.5;
        }
        .point-item {
            background: #16213e; padding: 10px; margin: 5px 0;
            border-radius: 4px; font-size: 12px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .point-item .info { flex: 1; }
        .point-item .name { font-weight: bold; color: #e94560; }
        .point-item .coords { color: #888; font-size: 11px; }
        .point-item .delete {
            background: #c0392b; padding: 4px 8px;
            border-radius: 3px; cursor: pointer; font-size: 11px;
        }

        .status {
            position: fixed; bottom: 20px; left: 20px;
            background: rgba(0,0,0,0.8); padding: 10px 15px;
            border-radius: 4px; font-size: 12px;
        }

        .zoom-controls {
            position: fixed; bottom: 20px; right: 370px;
            display: flex; gap: 5px;
        }
        .zoom-controls button { width: auto; padding: 8px 15px; }

        .instructions {
            background: #16213e; padding: 10px; border-radius: 4px;
            font-size: 11px; color: #aaa; margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="image-panel" id="image-panel">
            <div id="image-container">
                <img id="main-image" src="data:"""
        + mime_type
        + """;base64,"""
        + img_data
        + """">
                <svg id="connectors-svg"></svg>
            </div>
        </div>

        <div class="sidebar">
            <h2>KML Point Extractor</h2>

            <div class="instructions">
                <strong>Instructions:</strong><br>
                1. Select a category below<br>
                2. Click on image to place points<br>
                3. Work left to right<br>
                4. Export to KML when done
            </div>

            <div class="controls">
                <label>Category:</label>
                <select id="category">
                    <option value="zebra">Zebra Crossing Corner</option>
                    <option value="arrow">Arrow Tip</option>
                    <option value="parking">Parking Spot Corner</option>
                    <option value="other">Other</option>
                </select>

                <label>Point Name (auto-increments):</label>
                <input type="text" id="point-name" placeholder="e.g., Z1, A1, P1">

                <button onclick="exportKML()" class="secondary">Export KML</button>

                <label>Import KML:</label>
                <input type="file" id="kml-file" accept=".kml" onchange="importKML(this)">

                <button onclick="clearAll()">Clear All Points</button>
            </div>

            <div class="point-list">
                <h3>Points (<span id="point-count">0</span>)</h3>
                <div class="category-filters" id="category-filters">
                    <div class="category-btn zebra" onclick="toggleCategory('zebra')" data-category="zebra">Zebra</div>
                    <div class="category-btn arrow" onclick="toggleCategory('arrow')" data-category="arrow">Arrow</div>
                    <div class="category-btn parking" onclick="toggleCategory('parking')" data-category="parking">Parking</div>
                    <div class="category-btn other" onclick="toggleCategory('other')" data-category="other">Other</div>
                </div>
                <div id="points-container"></div>
            </div>
        </div>
    </div>

    <div class="status" id="status">Click on image to add points</div>

    <div class="zoom-controls">
        <button onclick="zoom(0.8)">-</button>
        <button onclick="zoom(1.25)">+</button>
        <button onclick="resetZoom()">Reset</button>
    </div>

    <script>
        const config = """
        + json.dumps(config)
        + """;
        let points = [];
        let currentZoom = 1;
        let counters = { zebra: 1, arrow: 1, parking: 1, other: 1 };
        let categoryVisibility = { zebra: true, arrow: true, parking: true, other: true };

        const img = document.getElementById('main-image');
        const container = document.getElementById('image-container');

        // Update point name based on category
        document.getElementById('category').addEventListener('change', updatePointName);
        updatePointName();

        function updatePointName() {
            const cat = document.getElementById('category').value;
            const prefix = { zebra: 'Z', arrow: 'A', parking: 'P', other: 'X' }[cat];
            document.getElementById('point-name').value = prefix + counters[cat];
        }

        container.addEventListener('click', function(e) {
            if (e.target !== img) return;

            const rect = img.getBoundingClientRect();
            const px = (e.clientX - rect.left) / currentZoom;
            const py = (e.clientY - rect.top) / currentZoom;

            const category = document.getElementById('category').value;
            const name = document.getElementById('point-name').value || (category + '_' + points.length);

            addPoint(px, py, name, category);

            // Increment counter
            counters[category]++;
            updatePointName();
        });

        function pixelToUTM(px, py) {
            // Apply GDAL 6-parameter affine geotransform
            // easting = GT[0] + px*GT[1] + py*GT[2]
            // northing = GT[3] + px*GT[4] + py*GT[5]
            const gt = config.geotransform;
            const easting = gt[0] + px * gt[1] + py * gt[2];
            const northing = gt[3] + px * gt[4] + py * gt[5];
            return { easting, northing };
        }

        function addPoint(px, py, name, category) {
            const utm = pixelToUTM(px, py);
            const point = { px, py, name, category, ...utm };
            points.push(point);

            // Redraw all markers to handle collision detection
            redrawMarkers();
            updatePointsList();
            updateStatus('Added: ' + name + ' at E:' + utm.easting.toFixed(2) + ' N:' + utm.northing.toFixed(2));
        }

        function updatePointsList() {
            const container = document.getElementById('points-container');
            container.innerHTML = points.map((p, i) => `
                <div class="point-item">
                    <div class="info">
                        <div class="name">${i+1}. ${p.name} (${p.category})</div>
                        <div class="coords">Pixel: (${p.px.toFixed(1)}, ${p.py.toFixed(1)})</div>
                        <div class="coords">UTM: E ${p.easting.toFixed(2)}, N ${p.northing.toFixed(2)}</div>
                    </div>
                    <div class="delete" onclick="deletePoint(${i})">X</div>
                </div>
            `).join('');
            document.getElementById('point-count').textContent = points.length;
        }

        function deletePoint(index) {
            points.splice(index, 1);
            redrawMarkers();
            updatePointsList();
        }

        function redrawMarkers() {
            // Remove all existing markers and dots
            document.querySelectorAll('.marker, .point-dot').forEach(m => m.remove());

            // Clear SVG connectors
            const svg = document.getElementById('connectors-svg');
            svg.innerHTML = '';

            // Update SVG size to match image
            const imgWidth = img.naturalWidth * currentZoom;
            const imgHeight = img.naturalHeight * currentZoom;
            svg.setAttribute('width', imgWidth);
            svg.setAttribute('height', imgHeight);

            // Get only visible points
            const visiblePoints = getVisiblePoints();

            // Calculate label positions with collision avoidance (only for visible points)
            const labelPositions = calculateLabelPositions(visiblePoints, imgWidth, imgHeight);

            visiblePoints.forEach((p, i) => {
                const pos = labelPositions[i];
                const pointX = p.px * currentZoom;
                const pointY = p.py * currentZoom;

                // Always draw a dot at the actual point location
                const dot = document.createElement('div');
                dot.className = 'point-dot ' + p.category;
                dot.style.left = pointX + 'px';
                dot.style.top = pointY + 'px';
                container.appendChild(dot);

                // Draw connector path if label is offset
                if (pos.offset && pos.path) {
                    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                    path.setAttribute('class', 'connector-path');
                    path.setAttribute('d', pos.path);
                    svg.appendChild(path);
                }

                // Draw the label
                const marker = document.createElement('div');
                marker.className = 'marker ' + p.category;
                marker.style.left = pos.x + 'px';
                marker.style.top = pos.y + 'px';
                marker.textContent = p.name;
                marker.dataset.index = points.indexOf(p);  // Use original index
                marker.onclick = (e) => { e.stopPropagation(); selectPoint(points.indexOf(p)); };
                container.appendChild(marker);
            });
        }

        function calculateLabelPositions(points, imgWidth, imgHeight) {
            const labelWidth = 35;  // Approximate label width
            const labelHeight = 18; // Approximate label height
            const minDistance = 28; // Minimum distance between label centers
            const offsetDistance = 45; // How far to offset colliding labels
            const padding = 5; // Padding from image edge

            // Initial positions (centered on point)
            const positions = points.map((p, i) => ({
                x: p.px * currentZoom,
                y: p.py * currentZoom,
                origX: p.px * currentZoom,
                origY: p.py * currentZoom,
                offset: false,
                path: null,
                index: i
            }));

            // Clamp position to stay within image bounds
            function clampToImage(x, y) {
                const halfW = labelWidth / 2;
                const halfH = labelHeight / 2;
                return {
                    x: Math.max(halfW + padding, Math.min(imgWidth - halfW - padding, x)),
                    y: Math.max(halfH + padding, Math.min(imgHeight - halfH - padding, y))
                };
            }

            // Create an elbow path from point to label
            function createElbowPath(px, py, lx, ly) {
                const dx = lx - px;
                const dy = ly - py;

                // Determine elbow style based on direction
                // Use horizontal-first or vertical-first depending on relative position
                let midX, midY;

                if (Math.abs(dx) > Math.abs(dy)) {
                    // Horizontal-first elbow
                    midX = lx;
                    midY = py;
                } else {
                    // Vertical-first elbow
                    midX = px;
                    midY = ly;
                }

                // If the elbow point would be very close to start or end, use direct line
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 30) {
                    return `M ${px} ${py} L ${lx} ${ly}`;
                }

                return `M ${px} ${py} L ${midX} ${midY} L ${lx} ${ly}`;
            }

            // Offset directions to try (8 directions around the point)
            const directions = [
                { dx: 1, dy: -1 },   // top-right
                { dx: 1, dy: 1 },    // bottom-right
                { dx: -1, dy: -1 },  // top-left
                { dx: -1, dy: 1 },   // bottom-left
                { dx: 1, dy: 0 },    // right
                { dx: -1, dy: 0 },   // left
                { dx: 0, dy: -1 },   // top
                { dx: 0, dy: 1 },    // bottom
            ];

            // Check for collisions and resolve them
            let iterations = 0;
            const maxIterations = 50;

            while (iterations < maxIterations) {
                let hasCollision = false;

                for (let i = 0; i < positions.length; i++) {
                    for (let j = i + 1; j < positions.length; j++) {
                        const dx = positions[i].x - positions[j].x;
                        const dy = positions[i].y - positions[j].y;
                        const dist = Math.sqrt(dx * dx + dy * dy);

                        if (dist < minDistance) {
                            hasCollision = true;

                            // Find best offset direction for the later point (j)
                            let bestDir = directions[0];
                            let bestScore = -Infinity;

                            for (const dir of directions) {
                                const rawX = positions[j].origX + dir.dx * offsetDistance * (1 + iterations * 0.2);
                                const rawY = positions[j].origY + dir.dy * offsetDistance * (1 + iterations * 0.2);
                                const clamped = clampToImage(rawX, rawY);

                                // Score based on:
                                // 1. Distance from all other labels (higher = better)
                                // 2. Staying close to original position (lower offset = better)
                                // 3. Penalty for being at the clamped edge
                                let score = 0;
                                for (let k = 0; k < positions.length; k++) {
                                    if (k !== j) {
                                        const d = Math.sqrt(
                                            Math.pow(clamped.x - positions[k].x, 2) +
                                            Math.pow(clamped.y - positions[k].y, 2)
                                        );
                                        score += d;
                                    }
                                }

                                // Slight penalty for large offsets
                                const offsetDist = Math.sqrt(
                                    Math.pow(clamped.x - positions[j].origX, 2) +
                                    Math.pow(clamped.y - positions[j].origY, 2)
                                );
                                score -= offsetDist * 0.1;

                                // Penalty if clamped (means we hit the edge)
                                if (clamped.x !== rawX || clamped.y !== rawY) {
                                    score -= 20;
                                }

                                if (score > bestScore) {
                                    bestScore = score;
                                    bestDir = { ...dir, clamped };
                                }
                            }

                            const rawX = positions[j].origX + bestDir.dx * offsetDistance * (1 + iterations * 0.2);
                            const rawY = positions[j].origY + bestDir.dy * offsetDistance * (1 + iterations * 0.2);
                            const finalPos = clampToImage(rawX, rawY);

                            positions[j].x = finalPos.x;
                            positions[j].y = finalPos.y;
                            positions[j].offset = true;
                        }
                    }
                }

                if (!hasCollision) break;
                iterations++;
            }

            // Generate paths for offset labels
            for (const pos of positions) {
                if (pos.offset) {
                    pos.path = createElbowPath(pos.origX, pos.origY, pos.x, pos.y);
                }
            }

            return positions;
        }

        function selectPoint(index) {
            updateStatus('Selected: ' + points[index].name);
        }

        function toggleCategory(category) {
            categoryVisibility[category] = !categoryVisibility[category];

            // Update button appearance
            const btn = document.querySelector(`.category-btn[data-category="${category}"]`);
            if (categoryVisibility[category]) {
                btn.classList.remove('hidden');
            } else {
                btn.classList.add('hidden');
            }

            // Redraw markers with new visibility
            redrawMarkers();
            updateStatus(category + ' labels ' + (categoryVisibility[category] ? 'shown' : 'hidden'));
        }

        function getVisiblePoints() {
            return points.filter(p => categoryVisibility[p.category]);
        }

        function zoom(factor) {
            currentZoom *= factor;
            img.style.width = (img.naturalWidth * currentZoom) + 'px';
            redrawMarkers();
        }

        function resetZoom() {
            currentZoom = 1;
            img.style.width = '';
            redrawMarkers();
        }

        function clearAll() {
            if (confirm('Clear all points?')) {
                points = [];
                counters = { zebra: 1, arrow: 1, parking: 1, other: 1 };
                redrawMarkers();
                updatePointsList();
                updatePointName();
            }
        }

        function exportKML() {
            if (points.length === 0) {
                alert('No points to export!');
                return;
            }

            fetch('/export', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ points: points })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    updateStatus('Exported to: ' + data.path);
                    alert('KML saved to: ' + data.path);
                }
            });
        }

        function importKML(input) {
            if (!input.files || !input.files[0]) return;

            const file = input.files[0];
            const reader = new FileReader();

            reader.onload = function(e) {
                const kmlText = e.target.result;

                fetch('/import', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ kml: kmlText })
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        // Clear existing points
                        points = [];
                        counters = { zebra: 1, arrow: 1, parking: 1, other: 1 };

                        // Add imported points
                        data.points.forEach(p => {
                            const utm = pixelToUTM(p.px, p.py);
                            points.push({
                                px: p.px,
                                py: p.py,
                                name: p.name,
                                category: p.category,
                                ...utm
                            });

                            // Update counters based on imported names
                            const cat = p.category;
                            const prefix = { zebra: 'Z', arrow: 'A', parking: 'P', other: 'X' }[cat] || 'X';
                            const match = p.name.match(new RegExp('^' + prefix + '(\\d+)$'));
                            if (match) {
                                counters[cat] = Math.max(counters[cat], parseInt(match[1]) + 1);
                            }
                        });

                        redrawMarkers();
                        updatePointsList();
                        updatePointName();
                        updateStatus('Imported ' + data.points.length + ' points from KML');
                        alert('Imported ' + data.points.length + ' points');
                    } else {
                        alert('Error importing KML: ' + data.error);
                    }
                });
            };

            reader.readAsText(file);
            input.value = ''; // Reset file input
        }

        function updateStatus(msg) {
            document.getElementById('status').textContent = msg;
        }
    </script>
</body>
</html>"""
    )


def run_server(image_path: str, geo_config: GeoConfig, port: int = 8765):
    """Run the web server."""

    extractor = PointExtractor(geo_config)
    # Convert to dict for JavaScript JSON serialization
    config_dict = {"crs": geo_config.crs, "geotransform": list(geo_config.geotransform)}
    html_content = create_html(image_path, config_dict)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(html_content.encode())
            else:
                self.send_error(404)

        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            post_data = json.loads(self.rfile.read(content_length))

            if self.path == "/export":
                # Clear and re-add points
                extractor.points = {}
                for p in post_data["points"]:
                    extractor.add_point(p["px"], p["py"], p["name"], p["category"])

                # Export
                output_path = str(Path(image_path).with_suffix(".kml"))
                kml_content = extractor.render_kml()
                with open(output_path, "w") as f:
                    f.write(kml_content)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"success": True, "path": output_path, "count": len(extractor.points)}
                    ).encode()
                )

            elif self.path == "/import":
                try:
                    kml_text = post_data.get("kml", "")
                    kml = Kml(kml_text)
                    imported = extractor.import_kml(kml.points)

                    # Convert to list format expected by frontend
                    points_list = [
                        {
                            "px": pixel.x,
                            "py": pixel.y,
                            "name": name,
                            "category": kml.category,
                        }
                        for name, (pixel, kml) in imported.items()
                    ]

                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "points": points_list}).encode())
                except Exception as e:
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass  # Suppress logging

    # Find available port
    port = find_available_port(start_port=port, max_attempts=10)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\n{'=' * 60}")
        print(f"KML Point Extractor running at: {url}")
        print(f"Image: {image_path}")
        print(f"CRS: {geo_config.crs}")
        print(f"Geotransform: {geo_config.geotransform}")
        print(f"{'=' * 60}")
        print("\nPress Ctrl+C to stop\n")

        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference points from georeferenced image to KML"
    )
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument(
        "--camera",
        type=str,
        default="Valte",
        help="Camera name to load configuration from (default: Valte)",
    )
    parser.add_argument(
        "--origin-e",
        type=float,
        default=None,
        help="Origin easting (UTM) - overrides camera config",
    )
    parser.add_argument(
        "--origin-n",
        type=float,
        default=None,
        help="Origin northing (UTM) - overrides camera config",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=None,
        help="Ground sample distance in meters - overrides camera config",
    )
    parser.add_argument(
        "--crs", default=None, help="Coordinate reference system - overrides camera config"
    )
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")

    args = parser.parse_args()

    # Load camera configuration (required - single source of truth)
    camera_config = get_camera_by_name(args.camera)

    if camera_config is None:
        print(f"Error: Camera '{args.camera}' not found in configuration.")
        print(f"Available cameras: {', '.join([c['name'] for c in get_camera_configs()])}")
        sys.exit(1)

    # Check if camera has geotiff_params
    if "geotiff_params" not in camera_config:
        print(f"Error: Camera '{args.camera}' does not have 'geotiff_params' defined.")
        print("Please update the camera configuration in poc_homography/camera_config.py")
        sys.exit(1)

    geotiff_params = camera_config["geotiff_params"]

    # Check for new geotransform format vs old format
    if "geotransform" in geotiff_params:
        # New format: use geotransform array directly
        gt = list(geotiff_params["geotransform"])
        crs = geotiff_params["utm_crs"]
        print(
            f"Loaded georeferencing parameters from camera: {args.camera} (new geotransform format)"
        )
    else:
        # Old format: build geotransform from separate parameters
        gt = [
            geotiff_params["origin_easting"],
            geotiff_params["pixel_size_x"],
            0.0,  # row_rotation (assumed 0 for legacy format)
            geotiff_params["origin_northing"],
            0.0,  # col_rotation (assumed 0 for legacy format)
            geotiff_params["pixel_size_y"],
        ]
        crs = geotiff_params["utm_crs"]
        print(
            f"Loaded georeferencing parameters from camera: {args.camera} (legacy format, converted to geotransform)"
        )

    # Command-line arguments override camera config
    if args.origin_e is not None or args.origin_n is not None or args.gsd is not None:
        # Override specific values
        if args.origin_e is not None:
            gt[0] = args.origin_e
        if args.origin_n is not None:
            gt[3] = args.origin_n
        if args.gsd is not None:
            gt[1] = args.gsd
            gt[5] = -args.gsd
        print("Applied command-line overrides to geotransform")

    if args.crs is not None:
        crs = args.crs

    geo_config = GeoConfig(
        crs=crs,
        geotransform=(gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]),
    )
    run_server(args.image, geo_config, args.port)


if __name__ == "__main__":
    main()
