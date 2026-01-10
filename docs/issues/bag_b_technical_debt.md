# Technical Debt Issues - Bag B

These issues are lower priority but should be addressed to improve code quality.

## 8. No input validation on GCP data
**Location:** webapp/gcp/views.py:95-97
**Problem:** No input validation on `data.get("gcps", [])` - the gcps list is accepted without any schema or type validation
**Recommendation:** Add JSON schema validation or use Django forms/serializers to validate input structure.

## 12. Empty models file
**Location:** webapp/gcp/models.py:1-2
**Problem:** Empty models file with no Django models defined
**Recommendation:** Either define models for GCP data or remove unused Django model infrastructure (admin, migrations).

## 15. API delete endpoint is deceptive
**Location:** webapp/gcp/views.py:125
**Problem:** API delete endpoint returns success without validating or actually deleting anything
**Recommendation:** Either implement real deletion or remove the endpoint until it's needed.

---
*Generated from PR #160 code review*
