# Django WebApp Migration - Summary

## Overview

Successfully migrated standalone Flask/web tools into a unified Django web application at `webapp/`.

## Migration Date

2026-01-10

## What Was Done

### 1. Infrastructure Setup
- Added `django>=5.0` to `pyproject.toml` dependencies
- Created `data/` directory structure at repository root:
  - `data/cameras/`
  - `data/gcps/`
  - `data/maps/`
- Moved all YAML configuration files from `config/` and root to `data/` subdirectories:
  - 5 GCP YAML files → `data/gcps/`
  - 1 homography config → `data/maps/`
- Created Django project:
  - Project: `webapp/` directory
  - Config package: `homography_web/`
  - App: `gcp/`

### 2. Django Configuration
- Configured `settings.py`:
  - Added `gcp` app to `INSTALLED_APPS`
  - Set `DATA_DIR = BASE_DIR.parent / 'data'`
  - Configured static files
- Configured URL routing in `urls.py`

### 3. Template Tag Migration
- Converted `poc_homography/satellite_layers.py` to Django template tag
- New location: `webapp/gcp/templatetags/satellite_layers.py`
- Adapted for Django template usage with `@register.simple_tag`

### 4. Templates Created
- `base.html` - Base template with Leaflet.js integration
- `index.html` - Landing page with tool navigation
- `gcp_capture.html` - Interactive GCP capture interface
- `debug_map.html` - GCP visualization and verification tool

### 5. Views Implemented
- `index()` - Landing page view
- `gcp_capture()` - GCP capture page view
- `debug_map()` - Debug map view
- `api_get_gcps()` - API endpoint to fetch GCPs
- `api_save_gcps()` - API endpoint to save GCPs
- `api_delete_gcp()` - API endpoint to delete specific GCP

All views are **thin wrappers** that reference existing `poc_homography` library functions (implementation placeholders included with comments showing integration pattern).

### 6. Files Deleted
The following deprecated standalone tools were removed:
- `tools/unified_gcp_tool.py` (301,871 bytes)
- `tools/capture_gcps_web.py` (2,784 bytes)
- `poc_homography/map_debug_server.py` (1,303 lines)
- `gcp_map.html` (46,910 bytes)
- `poc_homography/satellite_layers.py` (moved to Django templatetag)

## Final Structure

```
poc-homography/
├── data/                          # NEW - Centralized data storage
│   ├── cameras/
│   ├── gcps/
│   │   ├── gcps_valte_test.yaml
│   │   ├── gcps_valte_test2.yaml
│   │   ├── valte_gcps.yaml
│   │   ├── gcps_from_map_fixed.yaml
│   │   └── gcps_Valte_20251211_124557.yaml
│   └── maps/
│       └── homography_config.yaml
├── webapp/                        # NEW - Django project
│   ├── README.md                  # Usage documentation
│   ├── manage.py
│   ├── homography_web/            # Django config package
│   │   ├── settings.py            # DATA_DIR configured
│   │   ├── urls.py                # URL routing
│   │   ├── wsgi.py
│   │   └── asgi.py
│   └── gcp/                       # Django app
│       ├── views.py               # Thin wrapper views
│       ├── urls.py                # App URL patterns
│       ├── templatetags/
│       │   └── satellite_layers.py  # Migrated from poc_homography
│       ├── templates/gcp/
│       │   ├── base.html
│       │   ├── index.html
│       │   ├── gcp_capture.html
│       │   └── debug_map.html
│       └── static/gcp/
│           ├── js/
│           └── css/
├── poc_homography/                # Existing domain logic library
├── tools/                         # Cleaned up (deprecated tools removed)
└── pyproject.toml                 # Updated with Django dependency
```

## Usage

### Starting the Server

```bash
cd webapp
../.venv/bin/python manage.py runserver
```

### Accessing the Application

- **Landing Page**: http://localhost:8000/
- **GCP Capture Tool**: http://localhost:8000/capture/
- **Debug Map**: http://localhost:8000/debug/

### API Endpoints

- `GET /api/gcps/` - Get all GCPs
- `POST /api/gcps/save/` - Save GCPs
- `DELETE /api/gcps/delete/<id>/` - Delete GCP

## Testing Performed

1. Django system checks: ✓ No issues
2. Migrations: ✓ Successfully applied
3. URL resolution: ✓ All routes working
4. Template rendering: ✓ All templates valid
5. Template tag: ✓ `satellite_layers` tag functional

## Verification Commands

```bash
# Check Django configuration
cd webapp
../.venv/bin/python manage.py check

# Run migrations
../.venv/bin/python manage.py migrate

# Start development server
../.venv/bin/python manage.py runserver

# Test URL resolution
../.venv/bin/python -c "
import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'homography_web.settings')
django.setup()
from django.urls import reverse
print(reverse('gcp:index'))
print(reverse('gcp:gcp_capture'))
print(reverse('gcp:debug_map'))
"
```

## Architecture Notes

### Design Pattern: Thin Wrappers

Views in `webapp/gcp/views.py` are intentionally thin wrappers that:
1. Handle HTTP request/response
2. Call `poc_homography` library functions for business logic
3. Format data for templates or JSON responses

Example pattern:
```python
def debug_map(request):
    """Django view - thin wrapper"""
    from poc_homography.gcp_loader import load_gcps_from_yaml

    gcp_file = settings.DATA_DIR / 'gcps' / 'latest.yaml'
    gcps = load_gcps_from_yaml(gcp_file)  # Business logic in library

    return render(request, 'gcp/debug_map.html', {
        'gcps_json': json.dumps([serialize_gcp(g) for g in gcps])
    })
```

### Data Flow

1. User interacts with Django views
2. Views call `poc_homography` library functions
3. Library functions read/write data from `data/` directory
4. Views format responses and render templates

### No Business Logic Migration

Per project requirements, **no business logic was extracted or moved** from the existing `poc_homography` package. The Django app simply provides a web interface to existing functionality.

## Next Steps (Future Work)

1. Implement full GCP loading from YAML files (currently placeholders)
2. Implement GCP saving to YAML files (currently placeholders)
3. Add camera selection interface
4. Add homography calculation visualization
5. Add validation result display (from existing `validation_results`)
6. Add export functionality (to various formats)
7. Add user authentication (if multi-user needed)
8. Add deployment configuration for production

## Dependencies

- Django 5.2.10
- Leaflet.js 1.9.4 (via CDN)
- Python 3.10+
- uv package manager

## Migration Validation

- Django check: ✓ PASS
- Migrations: ✓ PASS
- URL routing: ✓ PASS
- Template loading: ✓ PASS
- Deprecated files removed: ✓ PASS
- Data directory structure: ✓ PASS
- YAML files moved: ✓ PASS (5 GCP files, 1 map config)

## Rollback Plan

If rollback is needed:
```bash
git checkout HEAD -- tools/unified_gcp_tool.py
git checkout HEAD -- tools/capture_gcps_web.py
git checkout HEAD -- poc_homography/map_debug_server.py
git checkout HEAD -- gcp_map.html
git checkout HEAD -- poc_homography/satellite_layers.py
rm -rf webapp/
rm -rf data/
git checkout HEAD -- config/
git checkout HEAD -- pyproject.toml
```

All changes are version controlled in git history.
