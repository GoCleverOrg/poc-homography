"""
Camera Frame Annotation Tool.

A simple web tool to annotate camera frames with GCP references.
Displays the frame, allows clicking to mark points, and generates
YAML annotations that can be copy-pasted into test cases.

Usage:
    hom annotate frame <image_path>
    uv run python -m poc_homography.tools.camera_annotator <image_path>
"""

from __future__ import annotations

import html
import http.server
import json
import mimetypes
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "homography" / "test_data"
DEFAULT_GCP_FILE = TEST_DATA_DIR / "Cartografia_valencia_gcps.yaml"
DEFAULT_ANNOTATIONS_FILE = TEST_DATA_DIR / "valte_annotations.yaml"


@dataclass
class CameraAnnotatorState:
    """State container for the camera annotator application.

    Follows Django-style state management pattern from webapp/point_picker/state.py.
    """

    image_path: Path
    gcps_file: Path
    annotations_file: Path
    image_filename: str

    def load_gcps(self) -> list[dict]:
        """Load GCP IDs from the registry file."""
        if not self.gcps_file.exists():
            print(f"Warning: GCP file not found: {self.gcps_file}")
            return []

        with open(self.gcps_file) as f:
            data = yaml.safe_load(f)

        # Handle empty or malformed YAML files
        if not data or not isinstance(data, dict):
            print(f"Warning: GCP file is empty or invalid: {self.gcps_file}")
            return []

        points = data.get("points", [])
        gcps = []
        for p in points:
            # Validate required keys exist
            if not all(k in p for k in ("id", "pixel_x", "pixel_y")):
                print(f"Warning: Skipping malformed GCP entry: {p}")
                continue
            gcps.append({"id": p["id"], "pixel_x": p["pixel_x"], "pixel_y": p["pixel_y"]})

        print(f"Loaded {len(gcps)} GCPs from {self.gcps_file}")
        return gcps

    def load_existing_annotations(self) -> list[dict]:
        """Load existing annotations for the current image from the annotations file."""
        if not self.annotations_file.exists():
            return []

        with open(self.annotations_file) as f:
            data = yaml.safe_load(f)

        # Handle empty or malformed YAML files
        if not data or not isinstance(data, dict):
            return []

        test_cases = data.get("test_cases", [])
        for tc in test_cases:
            # Match by image filename
            if tc.get("image") == self.image_filename:
                annotations = tc.get("annotations", [])
                print(f"Loaded {len(annotations)} existing annotations for {self.image_filename}")
                return annotations

        return []

    def switch_image(self, filename: str) -> bool:
        """Switch to a different image file.

        Args:
            filename: Name of the image file in TEST_DATA_DIR.

        Returns:
            True if successful, False if file doesn't exist or is outside TEST_DATA_DIR.
        """
        new_path = TEST_DATA_DIR / filename
        if not new_path.exists():
            return False

        # Security: Ensure resolved path is within TEST_DATA_DIR (defense-in-depth)
        resolved_path = new_path.resolve()
        if not resolved_path.is_relative_to(TEST_DATA_DIR.resolve()):
            return False

        self.image_path = resolved_path
        self.image_filename = filename
        return True


# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_available_images() -> list[str]:
    """Scan TEST_DATA_DIR for available image files.

    Returns:
        Sorted list of image filenames matching supported extensions.
    """
    if not TEST_DATA_DIR.exists():
        return []

    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(f.name for f in TEST_DATA_DIR.glob(f"*{ext}"))
        # Also check uppercase extensions
        images.extend(f.name for f in TEST_DATA_DIR.glob(f"*{ext.upper()}"))

    # Remove duplicates and sort
    return sorted(set(images))


# Module-level state (Django pattern from webapp/point_picker/state.py)
_state: CameraAnnotatorState | None = None


def initialize_state(
    image_path: Path,
    gcps_file: Path | None = None,
    annotations_file: Path | None = None,
) -> None:
    """Initialize the module-level state.

    Args:
        image_path: Path to the camera frame image.
        gcps_file: Path to GCPs YAML file (default: Cartografia_valencia_gcps.yaml).
        annotations_file: Path to annotations YAML file (default: valte_annotations.yaml).
    """
    global _state
    _state = CameraAnnotatorState(
        image_path=image_path.resolve(),
        gcps_file=(gcps_file or DEFAULT_GCP_FILE).resolve() if gcps_file else DEFAULT_GCP_FILE,
        annotations_file=(annotations_file or DEFAULT_ANNOTATIONS_FILE).resolve()
        if annotations_file
        else DEFAULT_ANNOTATIONS_FILE,
        image_filename=image_path.name,
    )


def get_state() -> CameraAnnotatorState:
    """Get the current application state."""
    if _state is None:
        raise RuntimeError("Application not initialized. Call initialize_state() first.")
    return _state


def create_html(image_filename: str) -> str:
    """Create the HTML interface."""
    # Escape filename to prevent XSS via malicious filenames
    safe_filename = html.escape(image_filename)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Camera Frame Annotator</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            display: flex;
            height: 100vh;
            background: #1a1a1a;
        }}

        /* Left panel - Image */
        #image-panel {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #222;
            overflow: hidden;
        }}

        #image-container {{
            flex: 1;
            overflow: auto;
            cursor: crosshair;
        }}

        #image-wrapper {{
            position: relative;
            display: inline-block;
        }}

        #frame-image {{
            display: block;
            max-width: none;
        }}

        /* Droplet-style markers (same as point-picker) */
        .marker {{
            position: absolute;
            width: 24px;
            height: 28px;
            margin-left: -12px;
            margin-top: -28px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 9px;
            font-weight: bold;
            color: #fff;
            pointer-events: none;
            filter: drop-shadow(0 2px 3px rgba(0,0,0,0.4));
        }}

        .marker::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 2px;
            width: 20px;
            height: 20px;
            border-radius: 50% 50% 50% 0;
            transform: rotate(-45deg);
            transform-origin: center center;
            border: 2px solid white;
            background: #2196f3;
        }}

        .marker::after {{
            content: '';
            position: absolute;
            bottom: -2px;
            left: 50%;
            width: 6px;
            height: 6px;
            margin-left: -3px;
            background: #2196f3;
            border: 2px solid white;
            border-radius: 50%;
        }}

        .marker span {{
            position: absolute;
            top: 1px;
            left: 0;
            width: 100%;
            text-align: center;
            z-index: 1;
        }}

        .marker.pending::before {{
            background: #ff9800;
        }}

        .marker.pending::after {{
            background: #ff9800;
            animation: pulse 1s infinite;
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}

        /* Right panel - Controls & Preview */
        #control-panel {{
            width: 400px;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            border-left: 1px solid #333;
        }}

        .panel-section {{
            padding: 16px;
            border-bottom: 1px solid #ddd;
        }}

        .panel-section h2 {{
            font-size: 14px;
            color: #333;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        #image-selector {{
            width: 100%;
            padding: 10px 12px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: #fff;
            cursor: pointer;
            outline: none;
        }}

        #image-selector:hover {{
            border-color: #2196f3;
        }}

        #image-selector:focus {{
            border-color: #2196f3;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.2);
        }}

        #filename {{
            font-size: 12px;
            color: #666;
            word-break: break-all;
            margin-bottom: 8px;
        }}

        /* GCP Input */
        #gcp-input-section {{
            display: none;
        }}

        #gcp-input-section.active {{
            display: block;
            background: #e3f2fd;
        }}

        #gcp-search {{
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #2196f3;
            border-radius: 6px;
            outline: none;
        }}

        #gcp-search:focus {{
            border-color: #1976d2;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.2);
        }}

        #gcp-dropdown {{
            max-height: 200px;
            overflow-y: auto;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 8px;
        }}

        .gcp-option {{
            padding: 10px 12px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
        }}

        .gcp-option:hover, .gcp-option.selected {{
            background: #e3f2fd;
        }}

        .gcp-option .id {{
            font-weight: 600;
            color: #1976d2;
        }}

        .gcp-option .coords {{
            font-size: 11px;
            color: #666;
        }}

        /* Annotations list */
        #annotations-list {{
            flex: 1;
            overflow-y: auto;
            padding: 16px;
        }}

        .annotation-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: #fff;
            border-radius: 4px;
            margin-bottom: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }}

        .annotation-item .info {{
            font-size: 13px;
        }}

        .annotation-item .gcp-id {{
            font-weight: 600;
            color: #1976d2;
        }}

        .annotation-item .pixel {{
            color: #666;
            font-size: 11px;
        }}

        .annotation-item .delete-btn {{
            background: #f44336;
            color: #fff;
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }}

        .annotation-item .delete-btn:hover {{
            background: #d32f2f;
        }}

        /* YAML Preview */
        #yaml-section {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 200px;
        }}

        #yaml-preview {{
            flex: 1;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
            line-height: 1.5;
            padding: 12px;
            background: #263238;
            color: #aed581;
            border: none;
            resize: none;
            white-space: pre;
            overflow: auto;
        }}

        #copy-btn {{
            padding: 12px;
            background: #4caf50;
            color: #fff;
            border: none;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
        }}

        #copy-btn:hover {{
            background: #43a047;
        }}

        #copy-btn.copied {{
            background: #2196f3;
        }}

        /* Instructions */
        #instructions {{
            padding: 12px;
            background: #fff3e0;
            font-size: 12px;
            color: #e65100;
        }}

        /* Coordinates display */
        #coords-display {{
            position: fixed;
            bottom: 16px;
            left: 16px;
            background: rgba(0,0,0,0.8);
            color: #fff;
            padding: 8px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div id="image-panel">
        <div id="image-container">
            <div id="image-wrapper">
                <img id="frame-image" src="/image" alt="Camera Frame">
            </div>
        </div>
    </div>

    <div id="control-panel">
        <div class="panel-section">
            <h2>Select Image</h2>
            <select id="image-selector">
                <!-- Options populated via JavaScript -->
            </select>
        </div>

        <div class="panel-section">
            <h2>Camera Frame</h2>
            <div id="filename">{safe_filename}</div>
            <div id="instructions">Click on the image to add annotation points. Type to filter GCPs.</div>
        </div>

        <div id="gcp-input-section" class="panel-section">
            <h2>Select GCP for Point</h2>
            <input type="text" id="gcp-search" placeholder="Type to filter GCPs..." autocomplete="off">
            <div id="gcp-dropdown"></div>
        </div>

        <div class="panel-section">
            <h2>Annotations (<span id="annotation-count">0</span>)</h2>
        </div>
        <div id="annotations-list"></div>

        <div id="yaml-section" class="panel-section">
            <h2>YAML Preview</h2>
            <textarea id="yaml-preview" readonly></textarea>
            <button id="copy-btn">Copy to Clipboard</button>
        </div>
    </div>

    <div id="coords-display">Move mouse over image</div>

    <script>
        // State
        let gcps = [];
        let annotations = [];
        let pendingPoint = null;
        let selectedGcpIndex = -1;
        let filteredGcps = [];

        // Elements
        const img = document.getElementById('frame-image');
        const wrapper = document.getElementById('image-wrapper');
        const gcpSection = document.getElementById('gcp-input-section');
        const gcpSearch = document.getElementById('gcp-search');
        const gcpDropdown = document.getElementById('gcp-dropdown');
        const annotationsList = document.getElementById('annotations-list');
        const annotationCount = document.getElementById('annotation-count');
        const yamlPreview = document.getElementById('yaml-preview');
        const copyBtn = document.getElementById('copy-btn');
        const coordsDisplay = document.getElementById('coords-display');
        const imageSelector = document.getElementById('image-selector');
        const filenameDisplay = document.getElementById('filename');

        // Load GCPs
        async function loadGcps() {{
            const resp = await fetch('/api/gcps');
            gcps = await resp.json();
            filteredGcps = [...gcps];
        }}

        // Render markers
        function renderMarkers() {{
            // Remove existing markers
            wrapper.querySelectorAll('.marker').forEach(m => m.remove());

            // Add annotation markers
            annotations.forEach((ann, i) => {{
                const marker = document.createElement('div');
                marker.className = 'marker';
                marker.style.left = ann.pixel_x + 'px';
                marker.style.top = ann.pixel_y + 'px';
                const label = document.createElement('span');
                label.textContent = ann.gcp_id;
                marker.appendChild(label);
                wrapper.appendChild(marker);
            }});

            // Add pending marker
            if (pendingPoint) {{
                const marker = document.createElement('div');
                marker.className = 'marker pending';
                marker.style.left = pendingPoint.x + 'px';
                marker.style.top = pendingPoint.y + 'px';
                const label = document.createElement('span');
                label.textContent = '?';
                marker.appendChild(label);
                wrapper.appendChild(marker);
            }}
        }}

        // Filter GCPs
        function filterGcps(query) {{
            const q = query.toLowerCase();
            filteredGcps = gcps.filter(g => g.id.toLowerCase().includes(q));

            // Exclude already used GCPs
            const usedIds = new Set(annotations.map(a => a.gcp_id));
            filteredGcps = filteredGcps.filter(g => !usedIds.has(g.id));

            selectedGcpIndex = filteredGcps.length > 0 ? 0 : -1;
            renderGcpDropdown();
        }}

        // Render GCP dropdown
        function renderGcpDropdown() {{
            gcpDropdown.innerHTML = '';

            if (filteredGcps.length === 0) {{
                gcpDropdown.innerHTML = '<div style="padding: 12px; color: #999;">No matching GCPs</div>';
                return;
            }}

            filteredGcps.forEach((gcp, i) => {{
                const div = document.createElement('div');
                div.className = 'gcp-option' + (i === selectedGcpIndex ? ' selected' : '');
                div.innerHTML = `
                    <span class="id">${{gcp.id}}</span>
                    <span class="coords">(${{gcp.pixel_x.toFixed(1)}}, ${{gcp.pixel_y.toFixed(1)}})</span>
                `;
                div.addEventListener('click', () => selectGcp(gcp));
                gcpDropdown.appendChild(div);
            }});
        }}

        // Select GCP for pending point
        function selectGcp(gcp) {{
            if (!pendingPoint) return;

            annotations.push({{
                pixel_x: pendingPoint.x,
                pixel_y: pendingPoint.y,
                gcp_id: gcp.id
            }});

            pendingPoint = null;
            gcpSection.classList.remove('active');
            gcpSearch.value = '';

            renderMarkers();
            renderAnnotationsList();
            updateYamlPreview();
        }}

        // Render annotations list
        function renderAnnotationsList() {{
            annotationsList.innerHTML = '';
            annotationCount.textContent = annotations.length;

            annotations.forEach((ann, i) => {{
                const div = document.createElement('div');
                div.className = 'annotation-item';
                div.innerHTML = `
                    <div class="info">
                        <span class="gcp-id">${{ann.gcp_id}}</span>
                        <span class="pixel">(${{ann.pixel_x.toFixed(1)}}, ${{ann.pixel_y.toFixed(1)}})</span>
                    </div>
                    <button class="delete-btn" data-index="${{i}}">Delete</button>
                `;
                annotationsList.appendChild(div);
            }});

            // Add delete handlers
            annotationsList.querySelectorAll('.delete-btn').forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    const idx = parseInt(e.target.dataset.index);
                    annotations.splice(idx, 1);
                    renderMarkers();
                    renderAnnotationsList();
                    updateYamlPreview();
                }});
            }});
        }}

        // Update YAML preview
        function updateYamlPreview() {{
            if (annotations.length === 0) {{
                yamlPreview.value = '# Click on image to add annotations';
                return;
            }}

            let yaml = 'annotations:\\n';
            annotations.forEach(ann => {{
                yaml += `  - pixel_x: ${{ann.pixel_x.toFixed(1)}}\\n`;
                yaml += `    pixel_y: ${{ann.pixel_y.toFixed(1)}}\\n`;
                yaml += `    gcp_id: ${{ann.gcp_id}}\\n\\n`;
            }});

            yamlPreview.value = yaml;
        }}

        // Event: Click on image
        img.addEventListener('click', (e) => {{
            const rect = img.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            pendingPoint = {{ x, y }};
            gcpSection.classList.add('active');
            gcpSearch.value = '';
            filterGcps('');
            gcpSearch.focus();

            renderMarkers();
        }});

        // Event: Mouse move for coordinates
        img.addEventListener('mousemove', (e) => {{
            const rect = img.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            coordsDisplay.textContent = `Pixel: (${{x.toFixed(1)}}, ${{y.toFixed(1)}})`;
        }});

        // Event: GCP search input
        gcpSearch.addEventListener('input', (e) => {{
            filterGcps(e.target.value);
        }});

        // Event: Keyboard navigation in dropdown
        gcpSearch.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowDown') {{
                e.preventDefault();
                if (selectedGcpIndex < filteredGcps.length - 1) {{
                    selectedGcpIndex++;
                    renderGcpDropdown();
                }}
            }} else if (e.key === 'ArrowUp') {{
                e.preventDefault();
                if (selectedGcpIndex > 0) {{
                    selectedGcpIndex--;
                    renderGcpDropdown();
                }}
            }} else if (e.key === 'Enter') {{
                e.preventDefault();
                if (selectedGcpIndex >= 0 && filteredGcps[selectedGcpIndex]) {{
                    selectGcp(filteredGcps[selectedGcpIndex]);
                }}
            }} else if (e.key === 'Escape') {{
                pendingPoint = null;
                gcpSection.classList.remove('active');
                renderMarkers();
            }}
        }});

        // Event: Copy button
        copyBtn.addEventListener('click', () => {{
            yamlPreview.select();
            document.execCommand('copy');
            copyBtn.textContent = 'Copied!';
            copyBtn.classList.add('copied');
            setTimeout(() => {{
                copyBtn.textContent = 'Copy to Clipboard';
                copyBtn.classList.remove('copied');
            }}, 2000);
        }});

        // Load existing annotations
        async function loadExistingAnnotations() {{
            try {{
                const resp = await fetch('/api/annotations');
                const existing = await resp.json();
                if (existing && existing.length > 0) {{
                    annotations = existing;
                    renderMarkers();
                    renderAnnotationsList();
                    updateYamlPreview();
                    console.log(`Loaded ${{existing.length}} existing annotations`);
                }}
            }} catch (e) {{
                console.error('Failed to load existing annotations:', e);
            }}
        }}

        // Load available images and populate selector
        async function loadAvailableImages() {{
            try {{
                const resp = await fetch('/api/images');
                const images = await resp.json();
                imageSelector.innerHTML = '';
                images.forEach(filename => {{
                    const option = document.createElement('option');
                    option.value = filename;
                    option.textContent = filename;
                    // Select current image
                    if (filename === filenameDisplay.textContent) {{
                        option.selected = true;
                    }}
                    imageSelector.appendChild(option);
                }});
                console.log(`Loaded ${{images.length}} available images`);
            }} catch (e) {{
                console.error('Failed to load available images:', e);
            }}
        }}

        // Switch to a different image
        async function switchImage(filename) {{
            console.log('switchImage called with:', filename);
            try {{
                const resp = await fetch('/api/switch-image', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ filename }})
                }});
                console.log('Response status:', resp.status);
                const data = await resp.json();
                console.log('Response data:', data);

                if (data.success) {{
                    // Clear pending point first
                    pendingPoint = null;
                    gcpSection.classList.remove('active');

                    // Load new annotations from response
                    annotations = data.annotations || [];

                    // Update filename display
                    filenameDisplay.textContent = data.filename;

                    // Update image src with cache-busting query param
                    // Use onload to re-render markers after image loads
                    const newSrc = '/image?t=' + Date.now();
                    console.log('Setting image src to:', newSrc);
                    img.onload = function() {{
                        console.log('Image loaded successfully');
                        renderMarkers();
                    }};
                    img.onerror = function() {{
                        console.error('Image failed to load');
                    }};
                    img.src = newSrc;

                    // Update other UI elements
                    renderAnnotationsList();
                    updateYamlPreview();

                    console.log(`Switched to image: ${{filename}}, ${{annotations.length}} annotations`);
                }} else {{
                    console.error('Failed to switch image:', data.error);
                    alert('Failed to switch image: ' + (data.error || 'Unknown error'));
                }}
            }} catch (e) {{
                console.error('Failed to switch image:', e);
                alert('Failed to switch image: ' + e.message);
            }}
        }}

        // Event: Image selector change
        imageSelector.addEventListener('change', (e) => {{
            switchImage(e.target.value);
        }});

        // Initialize
        loadGcps().then(() => {{
            loadAvailableImages().then(() => {{
                loadExistingAnnotations().then(() => {{
                    updateYamlPreview();
                }});
            }});
        }});
    </script>
</body>
</html>"""


class AnnotatorHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the annotator.

    Uses module-level state via get_state() following Django patterns.
    """

    def log_message(self, _format: str, *_args) -> None:
        """Suppress logging."""
        pass

    def _send_cors_headers(self) -> None:
        """Add CORS headers to restrict cross-origin access to localhost only."""
        origin = self.headers.get("Origin", "")
        # Only allow localhost origins to prevent cross-site data exfiltration
        if origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:"):
            self.send_header("Access-Control-Allow-Origin", origin)
        # If no origin header (same-origin request), don't add CORS headers

    def _send_json_response(self, data: list | dict, status: int = 200) -> None:
        """Send a JSON response with proper headers."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_json_error(self, message: str, status: int = 500) -> None:
        """Send a JSON error response."""
        self._send_json_response({"error": message}, status)

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Get state via Django-style accessor
        try:
            state = get_state()
        except RuntimeError as e:
            self.send_error(500, str(e))
            return

        if path == "/":
            # Serve HTML
            html_content = create_html(state.image_filename)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode())

        elif path == "/image":
            # Serve the image with no-cache headers to ensure fresh content after switch
            if state.image_path.exists():
                mime_type, _ = mimetypes.guess_type(str(state.image_path))
                self.send_response(200)
                self.send_header("Content-Type", mime_type or "image/jpeg")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                with open(state.image_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Image not found")

        elif path == "/api/gcps":
            # Serve GCP list with error handling
            try:
                gcps = state.load_gcps()
                self._send_json_response(gcps)
            except Exception as e:
                self._send_json_error(f"Failed to load GCPs: {e}")

        elif path == "/api/annotations":
            # Serve existing annotations with error handling
            try:
                annotations = state.load_existing_annotations()
                self._send_json_response(annotations)
            except Exception as e:
                self._send_json_error(f"Failed to load annotations: {e}")

        elif path == "/api/images":
            # Serve list of available images
            try:
                images = get_available_images()
                self._send_json_response(images)
            except Exception as e:
                self._send_json_error(f"Failed to get available images: {e}")

        else:
            self.send_error(404)

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Get state via Django-style accessor
        try:
            state = get_state()
        except RuntimeError as e:
            self._send_json_error(str(e), 500)
            return

        if path == "/api/switch-image":
            # Switch to a different image
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())

                filename = data.get("filename", "")

                # Security: Validate filename to prevent path traversal
                if "/" in filename or ".." in filename or not filename:
                    self._send_json_error("Invalid filename", 400)
                    return

                # Attempt to switch image
                if state.switch_image(filename):
                    # Load annotations for the new image
                    annotations = state.load_existing_annotations()
                    self._send_json_response(
                        {
                            "success": True,
                            "filename": filename,
                            "annotations": annotations,
                        }
                    )
                else:
                    self._send_json_error(f"Image not found: {filename}", 404)

            except json.JSONDecodeError:
                self._send_json_error("Invalid JSON", 400)
            except Exception as e:
                self._send_json_error(f"Failed to switch image: {e}", 500)

        else:
            self.send_error(404)


def run_annotator(
    image_path: Path | None = None, gcps_file: Path | None = None, port: int = 8888
) -> None:
    """Run the annotation server.

    Args:
        image_path: Path to the image to annotate. If None, starts in selector mode
                   using the first available image from TEST_DATA_DIR.
        gcps_file: Path to GCPs YAML file.
        port: Port to run the server on.
    """
    # Handle selector mode (image_path is None)
    if image_path is None:
        available_images = get_available_images()
        if not available_images:
            print("Error: No images found in test data directory.")
            print(f"Directory: {TEST_DATA_DIR}")
            print("Supported extensions: .jpg, .jpeg, .png")
            sys.exit(1)

        # Use first image alphabetically
        image_path = TEST_DATA_DIR / available_images[0]
        print(f"Starting in selector mode with {len(available_images)} images available")
    else:
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)

    # Initialize state using Django-style pattern
    initialize_state(image_path, gcps_file=gcps_file)
    state = get_state()

    try:
        server = http.server.HTTPServer(("localhost", port), AnnotatorHandler)
    except OSError as e:
        if "Address already in use" in str(e) or e.errno == 48:
            print(f"Error: Port {port} is already in use.")
            print(f"Try a different port with: hom annotate frame {image_path} --port <port>")
            sys.exit(1)
        raise

    # Load existing annotations count
    existing = state.load_existing_annotations()

    print("Camera Frame Annotator")
    print("=" * 50)
    print(f"Image: {state.image_path}")
    print(f"GCPs:  {state.gcps_file}")
    print(f"Existing annotations: {len(existing)}")
    print("")
    print(f"Server running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print()

    # Open browser
    def open_browser():
        import time

        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m poc_homography.tools.camera_annotator <image_path>")
        print()
        print("Example:")
        print(
            "  python -m poc_homography.tools.camera_annotator tests/homography/test_data/valte_102.5_20.7_1_20260115_112639.jpg"
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])
    run_annotator(image_path)


if __name__ == "__main__":
    main()
