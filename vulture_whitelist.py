# ===========================================================================
# Vulture whitelist — symbols that are unused in production code but are
# intentionally kept.  Each entry is categorised:
#
#   backward-compat     Alias kept for downstream/notebook consumers
#   public API          Method/property on a public class; called by tests,
#                       notebooks or external consumers only
#   CLI entry point     Typer/Click decorated command registered at import time
#   dataclass/NT field  Dataclass or NamedTuple field (accessed dynamically)
#   numpy immutability  ndarray.flags.writeable = False (numpy internal)
#   enum member         Enum value not referenced in scanned code
#   protocol iface      Protocol / ABC method; implementations are the callers
#   class constant      Class-level constant consumed at runtime via self.*
#   interface impl      Concrete method fulfilling an abstract interface
#   Django boilerplate  Django-required module-level or class-level symbol
#   Django template tag Template tag/filter loaded by {% load ... %}
#   dead local variable Assigned but never read — kept to document intent
#   uncertain           Not clearly called; kept conservatively — see note
# ===========================================================================

# ── poc_homography.calibration ─────────────────────────────────────────────
DistortionCoefficients  # backward-compat — alias for LensDistortion
_.remove_entry  # public API — CalibrationTable method; tested in test_calibration_table.py
_.get_zoom_levels  # public API — CalibrationTable method; tested in test_calibration_table.py
_.get_nearest_entry  # public API — CalibrationTable method (no production callers, only tests)
load_calibration_for_camera  # public API — standalone loader; tested in test_calibration_table.py
_.get_coefficients  # public API — CalibrationTable interpolation; tested in test_calibration_table.py
_.get_entry  # public API — CalibrationTable exact lookup (no prod callers, only used via get_coefficients)
merge_distance_threshold  # dataclass/NT field — LineDetectionParams field used by OpenCV pipeline
_.length_pixels  # public API — DetectedLine property; tested in test_lens_distortion_models.py
_.angle_degrees  # public API — DetectedLine property; tested in test_lens_distortion_models.py
_.to_points_array  # public API — DetectedLine method (no callers, but part of model interface)
_.length_meters  # public API — GroundTruthLine property; tested in test_lens_distortion_models.py
_.to_points_array  # public API — GroundTruthLine method (no callers, but part of model interface)
ground_truth_line  # dataclass/NT field — LineCorrespondence field; tested
_.to_world_coordinates  # public API — DetectedLine method; tested in test_lens_distortion_models.py
error_pixels  # dataclass/NT field — ProjectionAnalysisResult field
best_joint_pan_offset  # dataclass/NT field — ProjectionAnalysisResult field

# ── poc_homography.camera_geometry ─────────────────────────────────────────
_.project_image_to_world  # public API — CameraGeometry projection (no prod callers)
_.project_world_to_image  # public API — CameraGeometry projection (no prod callers)
_.world_to_map  # public API — CameraGeometry conversion (no prod callers)
_.undistort_point  # public API — CameraGeometry distortion (no prod callers)

# ── numpy immutability (ndarray.flags.writeable = False) ───────────────────
_.writeable  # numpy immutability — CameraParameters (camera_parameters.py)
_.writeable  # numpy immutability — CameraParameters (camera_parameters.py)
_.writeable  # numpy immutability — CameraParameters (camera_parameters.py)
_.writeable  # numpy immutability — CameraParameters (camera_parameters.py)
_.writeable  # numpy immutability — CameraParameters (camera_parameters.py)

# ── CLI entry points (typer-decorated, discovered at import) ───────────────
projection_command  # CLI entry point — `poe calibrate projection`
comprehensive_command  # CLI entry point — `poe calibrate comprehensive`
YAML  # enum member — OutputFormat.YAML in cli/camera.py
intrinsics_command  # CLI entry point — `poe camera intrinsics`
validate_command  # CLI entry point — `poe camera validate`
interactive_command  # CLI entry point — `poe interactive`
serve  # CLI entry point — line-picker serve
serve  # CLI entry point — point-picker serve
sam3_command  # CLI entry point — `poe test sam3`

# ── poc_homography.domain ──────────────────────────────────────────────────
HIKVISION_DS_2DF8425IX  # enum member — CameraSpec; only enum variant, used via from_model_string()
_.last_ptz_state  # protocol iface — CameraController protocol property
_.move_relative  # protocol iface — CameraController protocol method
_.writeable  # numpy immutability — Matrix3x3 value object
_.from_matrix  # public API — Rotation classmethod (no prod callers)
_.to_matrix  # public API — Rotation method (no prod callers)
_.writeable  # numpy immutability — Vector3 value object
_.normalized  # public API — Vector3 method (no prod callers)
_.focal_length_at_zoom  # public API — CameraSpec method (no prod callers)
_.area  # public API — ImageDimensions property (no prod callers)
_.center_x  # public API — ImageDimensions property (no prod callers)
_.center_y  # public API — ImageDimensions property (no prod callers)
_.has_distortion  # public API — LensDistortion property (no prod callers)
_.inverse  # public API — Matrix3x3 method (no prod callers)
_.to_orientation  # public API — PTZState method (no prod callers)

# ── poc_homography.homography.config ───────────────────────────────────────
_.get_approach_config  # public API — HomographyConfig method; tested in test_gcp_validation.py
_.save_to_yaml  # public API — HomographyConfig method (no prod callers)

# ── poc_homography.homography.interface ────────────────────────────────────
FEATURE_MATCH  # enum member — HomographyApproach; used in tests
LEARNED  # enum member — HomographyApproach; unused but part of enum
MAP_BASED_ORIGIN  # enum member — HomographyType; unused but part of enum
_.identity  # public API — HomographyResult classmethod (no prod callers)
_.inverse  # public API — HomographyResult method (no prod callers)
_.transform  # public API — HomographyResult method (no prod callers)
_.project_point  # protocol iface — AbstractHomography abstract method
_.project_points  # protocol iface — AbstractHomography abstract method
image_points  # dead local variable — abstract method parameter name
point_id_prefix  # dead local variable — abstract method parameter name
_.get_confidence  # protocol iface — AbstractHomography abstract method

# ── poc_homography.homography.intrinsic_extrinsic ──────────────────────────
EDGE_FACTOR_CENTER  # class constant — IntrinsicExtrinsicHomography confidence weighting
EDGE_FACTOR_EDGE  # class constant — IntrinsicExtrinsicHomography confidence weighting
EDGE_FACTOR_CORNER_DECAY  # class constant — IntrinsicExtrinsicHomography confidence weighting
EDGE_FACTOR_MIN  # class constant — IntrinsicExtrinsicHomography confidence weighting
_.get_distortion_coefficients  # public API — tested in test_calibration_table.py
_.compute_from_config  # public API — core classmethod; tested extensively
_.project_point  # interface impl — AbstractHomography (deprecated, raises)
_.project_points  # interface impl — AbstractHomography (deprecated, raises)
_.get_confidence  # interface impl — AbstractHomography (deprecated, raises)
_.project_point_static  # public API — static projection (no prod callers)
_.project_image_to_map  # public API — batch projection (no prod callers)
_.result_to_homography_result  # public API — converter (no prod callers)
_.load_lens_distortion_calibration  # public API — calibration loader (no prod callers)
_.merge_calibration_tables  # public API — calibration merger (no prod callers)

# ── poc_homography.homography.map_points ───────────────────────────────────
inverse_matrix  # dataclass/NT field — MapPointComputationResult field
inverse_matrix  # dataclass/NT field — LineComputationResult field
_._point_counter  # uncertain — MapPointHomography internal counter; set but never read
#   NOTE: _point_counter is assigned in __init__ but never referenced elsewhere.
#   May be vestigial from a removed feature, but kept as it's harmless.
iteration  # dead local variable — RANSAC loop variable (intent documentation)
_.camera_to_map_batch  # public API — batch projection; tested in test_map_points_integration.py
_.map_to_camera_batch  # public API — batch projection; tested in test_map_points_integration.py
_.get_result  # public API — MapPointHomography accessor (no prod callers)
_.get_homography_matrix  # public API — tested in test_map_points_integration.py
_.get_inverse_matrix  # public API — tested in test_map_points_integration.py

# ── poc_homography.homography.parameters ───────────────────────────────────
_.writeable  # numpy immutability — HomographyParameters
_.writeable  # numpy immutability — HomographyParameters
_.from_reference_dict  # public API — IntrinsicExtrinsicConfig classmethod; tested
_.to_reference_dict  # public API — IntrinsicExtrinsicConfig method (no prod callers)
_.writeable  # numpy immutability — HomographyMapParameters
_.writeable  # numpy immutability — HomographyMapParameters
_.writeable  # numpy immutability — HomographyMapParameters
_.map_dimensions  # public API — HomographyMapParameters property; tested in test_gcp_validation.py

# ── poc_homography.infrastructure ──────────────────────────────────────────
_.last_ptz_state  # protocol iface — HikvisionCameraController implementing CameraController
_.move_relative  # protocol iface — HikvisionCameraController implementing CameraController
_.get_by_map  # public API — MixinRepoMapFilter mixin method
_.clear_cache  # public API — RepoYaml cache invalidation
_.get_by_map  # public API — RepoYamlCapturedFrame.get_by_map()
_.get_by_camera  # public API — RepoYamlCapturedFrame.get_by_camera()
_.clear_cache  # public API — RepoYamlCapturedFrame.clear_cache()
_.image_dir_for  # public API — RepoYamlCapturedFrame; tested in test_repo_yaml_captured_frame.py

# ── webapp — Django boilerplate (apps.py / urls.py / settings.py) ──────────
CameraAnnotatorConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
CameraDiagnosticConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
CameraEvaluationConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
CameraLineAnnotatorConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
CameraSurveyConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
GcpConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
default_app_config  # Django boilerplate — homography_precision __init__.py
HomographyPrecisionConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
application  # Django boilerplate — ASGI application
SECRET_KEY  # Django boilerplate — settings.py
ALLOWED_HOSTS  # Django boilerplate — settings.py
INSTALLED_APPS  # Django boilerplate — settings.py
MIDDLEWARE  # Django boilerplate — settings.py
ROOT_URLCONF  # Django boilerplate — settings.py
TEMPLATES  # Django boilerplate — settings.py
WSGI_APPLICATION  # Django boilerplate — settings.py
DATABASES  # Django boilerplate — settings.py
AUTH_PASSWORD_VALIDATORS  # Django boilerplate — settings.py
LANGUAGE_CODE  # Django boilerplate — settings.py
TIME_ZONE  # Django boilerplate — settings.py
USE_I18N  # Django boilerplate — settings.py
USE_TZ  # Django boilerplate — settings.py
STATIC_URL  # Django boilerplate — settings.py
STATICFILES_DIRS  # Django boilerplate — settings.py
DEFAULT_AUTO_FIELD  # Django boilerplate — settings.py
urlpatterns  # Django boilerplate — URL routing (homography_web/urls.py)
application  # Django boilerplate — WSGI application
LensCalibrationConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
LinePickerConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
default_app_config  # Django boilerplate — point_picker __init__.py
PointPickerConfig  # Django boilerplate — AppConfig class
default_auto_field  # Django boilerplate — AppConfig attribute
verbose_name  # Django boilerplate — AppConfig attribute
app_name  # Django boilerplate — URL namespace
urlpatterns  # Django boilerplate — URL routing
DATA_DIR  # Django boilerplate — settings.py custom constant

# ── webapp — diagnostic models / template tags ─────────────────────────────
_.get_overall_status  # uncertain — DiagnosticResult method; not called from templates or views
#   NOTE: get_overall_status is defined but never invoked. Likely intended for
#   a diagnostic summary page that hasn't been built yet.
GOOD  # enum member — DiagnosticTestStatus (may be used in templates via string comparison)
NEEDS_IMPROVEMENT  # enum member — DiagnosticTestStatus (may be used in templates via string comparison)
BAD  # enum member — DiagnosticTestStatus (may be used in templates via string comparison)
VALID_LAYERS  # uncertain — satellite_layers template tag constant
#   NOTE: VALID_LAYERS is defined but the satellite_layers template tag is not
#   loaded in any .html template. May be unused dead code.
satellite_layers_js  # uncertain — template tag function
#   NOTE: satellite_layers_js template tag is not {% load %}ed in any template.
#   May be dead code from an earlier GCP picker design.

# ── webapp — views and middleware ──────────────────────────────────────────
_.modified  # Django boilerplate — request.session.modified = True (Django session API)
camera_model  # dead local variable — create_ptz_camera parameter (future: brand detection)
homography_source  # dead local variable — assigned but never read in api_compute_line_errors
line_result  # dead local variable — compute_from_lines result not used after assignment
TenantIdMiddleware  # Django boilerplate — referenced in MIDDLEWARE setting
_.process_exception  # Django boilerplate — Django middleware hook called by framework
validate_add_line_request  # uncertain — validation function with no callers
#   NOTE: validate_add_line_request() is defined in line_picker/validation.py but
#   is never imported or called by views.py. May be dead code from an earlier API design.
validate_export_request  # uncertain — validation function with no callers
#   NOTE: validate_export_request() is defined but never imported by views.py.
#   The export endpoint may do inline validation instead.
validate_import_request  # uncertain — validation function with no callers
#   NOTE: validate_import_request() is defined but never imported by views.py.
#   The import endpoint may do inline validation instead.
