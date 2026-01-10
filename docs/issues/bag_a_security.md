# Security Issues - Bag A

These security issues should be addressed before production deployment.

## 1. Hardcoded insecure SECRET_KEY
**Location:** webapp/homography_web/settings.py:26
**Problem:** Hardcoded SECRET_KEY with insecure prefix: `SECRET_KEY = "django-insecure-33xd23v-g5kb1@!t-y3%10m8rx#_7&7jkoqey%=*vo&99l6gv)"`
**Recommendation:** Use environment variable `os.environ.get('DJANGO_SECRET_KEY')` with fallback for development only.

## 2. DEBUG mode hardcoded
**Location:** webapp/homography_web/settings.py:29
**Problem:** `DEBUG = True` is hardcoded with no environment-based override
**Recommendation:** Use `DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() == 'true'`

## 3. Empty ALLOWED_HOSTS
**Location:** webapp/homography_web/settings.py:31
**Problem:** `ALLOWED_HOSTS = []` is empty
**Recommendation:** Configure based on environment: `ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')`

## 4. CSRF disabled on api_save_gcps
**Location:** webapp/gcp/views.py:82-83
**Problem:** `@csrf_exempt` decorator on `api_save_gcps` which handles POST requests that modify data
**Recommendation:** Remove `@csrf_exempt` and ensure frontend sends CSRF token.

## 5. CSRF disabled on api_delete_gcp
**Location:** webapp/gcp/views.py:110-111
**Problem:** `@csrf_exempt` decorator on `api_delete_gcp` which handles DELETE requests
**Recommendation:** Remove `@csrf_exempt` and ensure frontend sends CSRF token.

## 6. Exception details exposed in api_save_gcps
**Location:** webapp/gcp/views.py:106-107
**Problem:** Exception details exposed to client: `return JsonResponse({"success": False, "error": str(e)}, status=500)`
**Recommendation:** Log exception details server-side, return generic error message to client.

## 7. Exception details exposed in api_delete_gcp
**Location:** webapp/gcp/views.py:127-128
**Problem:** Exception details exposed to client: `return JsonResponse({"success": False, "error": str(e)}, status=500)`
**Recommendation:** Log exception details server-side, return generic error message to client.

---
*Generated from PR #160 code review*
