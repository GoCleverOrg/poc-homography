# Homography WebApp

Django-based web application for GCP (Ground Control Point) capture and visualization.

## Quick Start

1. Make sure dependencies are installed:
   ```bash
   cd ..  # Go to repository root
   uv sync
   ```

2. Run the Django development server:
   ```bash
   cd webapp
   ../.venv/bin/python manage.py runserver
   ```

3. Open your browser to:
   - http://localhost:8000/ - Landing page with tool links
   - http://localhost:8000/capture/ - GCP Capture Tool
   - http://localhost:8000/debug/ - Debug Map Visualization

## Project Structure

```
webapp/
├── manage.py                    # Django management script
├── homography_web/              # Django project configuration
│   ├── settings.py              # Settings (DATA_DIR points to ../data/)
│   └── urls.py                  # Main URL routing
└── gcp/                         # GCP app
    ├── views.py                 # View functions (thin wrappers)
    ├── urls.py                  # App URL patterns
    ├── templates/gcp/           # HTML templates
    │   ├── base.html
    │   ├── index.html
    │   ├── gcp_capture.html
    │   └── debug_map.html
    ├── templatetags/            # Django template tags
    │   └── satellite_layers.py  # Leaflet.js layer configuration
    └── static/gcp/              # Static files (CSS, JS)
        ├── css/
        └── js/
```

## Features

### GCP Capture Tool (`/capture/`)
- Interactive Leaflet.js map with satellite imagery
- Click-to-add GCP markers
- Support for multiple satellite layer providers:
  - ESRI Satellite
  - PNOA Spain (high-resolution orthophotos)
  - Google Satellite
  - OSM Street Map
  - Hybrid (satellite + streets)
- Save/load GCPs to data directory
- Real-time GCP count and management

### Debug Map Visualization (`/debug/`)
- Load existing GCPs from YAML files
- Visualize on satellite map for verification
- Display GCP metadata (coordinates, pixel positions)
- Useful for debugging homography calculations

## API Endpoints

- `GET /api/gcps/` - Get all GCPs
- `POST /api/gcps/save/` - Save GCPs to file
- `DELETE /api/gcps/delete/<id>/` - Delete specific GCP

## Data Directory

The webapp reads/writes data to `../data/` (repository root):
- `data/gcps/` - GCP YAML files
- `data/maps/` - Map configuration files
- `data/cameras/` - Camera data files

## Development

The views in `gcp/views.py` are thin wrappers that reference the `poc_homography` library.
The actual business logic lives in the `poc_homography` package at the repository root.

Example:
```python
# In views.py (thin wrapper)
from poc_homography.gcp_loader import load_gcps_from_yaml
from django.conf import settings

def debug_map(request):
    gcp_file = settings.DATA_DIR / 'gcps' / 'latest.yaml'
    gcps = load_gcps_from_yaml(gcp_file)
    # ... render template with gcps
```

## Technology Stack

- **Django 5.2+** - Web framework
- **Leaflet.js 1.9.4** - Interactive maps
- **Python 3.10+** - Runtime
- **uv** - Package manager

## Notes

- This is a development-only setup (DEBUG=True)
- No production deployment configuration included
- CSRF protection enabled for POST/DELETE endpoints
- Static files served via Django's development server
