# ruff: noqa: B018  # bare-name references are intentional in a vulture whitelist
DistortionCoefficients  # backward-compat alias (poc_homography/camera_parameters.py:21)
_.remove_entry  # unused method (poc_homography/calibration/lens_distortion/calibration_table.py:202)
_.get_zoom_levels  # unused method (poc_homography/calibration/lens_distortion/calibration_table.py:373)
_.get_nearest_entry  # unused method (poc_homography/calibration/lens_distortion/calibration_table.py:393)
merge_distance_threshold  # unused variable (poc_homography/calibration/lens_distortion/line_detection.py:61)
_.length_pixels  # unused property (poc_homography/calibration/lens_distortion/models.py:76)
_.angle_degrees  # unused property (poc_homography/calibration/lens_distortion/models.py:83)
_.to_points_array  # unused method (poc_homography/calibration/lens_distortion/models.py:90)
_.length_meters  # unused property (poc_homography/calibration/lens_distortion/models.py:213)
_.to_points_array  # unused method (poc_homography/calibration/lens_distortion/models.py:220)
ground_truth_line  # unused variable (poc_homography/calibration/lens_distortion/models.py:266)
_.to_world_coordinates  # unused method (poc_homography/calibration/lens_distortion/models.py:288)
error_pixels  # unused variable (poc_homography/calibration/projection.py:34)
best_joint_pan_offset  # unused variable (poc_homography/calibration/projection.py:43)
_.project_image_to_world  # unused method (poc_homography/camera_geometry.py:564)
_.project_world_to_image  # unused method (poc_homography/camera_geometry.py:595)
_.world_to_map  # unused method (poc_homography/camera_geometry.py:623)
_.undistort_point  # unused method (poc_homography/camera_geometry.py:652)
_.writeable  # unused attribute (poc_homography/camera_parameters.py:162)
_.writeable  # unused attribute (poc_homography/camera_parameters.py:173)
_.writeable  # unused attribute (poc_homography/camera_parameters.py:191)
_.writeable  # unused attribute (poc_homography/camera_parameters.py:338)
_.writeable  # unused attribute (poc_homography/camera_parameters.py:349)
projection_command  # unused function (poc_homography/cli/calibrate.py:21)
comprehensive_command  # unused function (poc_homography/cli/calibrate.py:75)
YAML  # unused variable (poc_homography/cli/camera.py:30)
intrinsics_command  # unused function (poc_homography/cli/camera.py:33)
validate_command  # unused function (poc_homography/cli/camera.py:210)
interactive_command  # unused function (poc_homography/cli/interactive.py:14)
serve  # unused function (poc_homography/cli/line_picker.py:17)
serve  # unused function (poc_homography/cli/point_picker.py:17)
sam3_command  # unused function (poc_homography/cli/test_cmds.py:14)
HIKVISION_DS_2DF8425IX  # unused variable (poc_homography/domain/enums/camera_spec.py:60)
_.last_ptz_state  # unused property (poc_homography/domain/protocols/camera_controller.py:24)
_.move_relative  # unused method (poc_homography/domain/protocols/camera_controller.py:67)
_.writeable  # unused attribute (poc_homography/domain/vo/matrix3x3.py:106)
_.from_matrix  # unused method (poc_homography/domain/vo/rotation.py:73)
_.to_matrix  # unused method (poc_homography/domain/vo/rotation.py:88)
_.writeable  # unused attribute (poc_homography/domain/vo/vector3.py:85)
_.normalized  # unused method (poc_homography/domain/vo/vector3.py:103)
_.get_approach_config  # unused method (poc_homography/homography/config.py:270)
_.save_to_yaml  # unused method (poc_homography/homography/config.py:290)
FEATURE_MATCH  # unused variable (poc_homography/homography/interface.py:37) - enum value used in tests
LEARNED  # unused variable (poc_homography/homography/interface.py:38)
MAP_BASED_ORIGIN  # unused variable (poc_homography/homography/interface.py:45)
_.identity  # unused method (poc_homography/homography/interface.py:89)
_.project_point  # unused method (poc_homography/homography/interface.py:205) - abstract interface method
_.project_points  # unused method (poc_homography/homography/interface.py:218)
image_points  # unused variable (poc_homography/homography/interface.py:220) - abstract method parameter
point_id_prefix  # unused variable (poc_homography/homography/interface.py:220) - abstract method parameter
_.get_confidence  # unused method (poc_homography/homography/interface.py:233)
EDGE_FACTOR_CENTER  # unused variable (poc_homography/homography/intrinsic_extrinsic.py:119) - class constant
EDGE_FACTOR_EDGE  # unused variable (poc_homography/homography/intrinsic_extrinsic.py:120) - class constant
EDGE_FACTOR_CORNER_DECAY  # unused variable (poc_homography/homography/intrinsic_extrinsic.py:121)
EDGE_FACTOR_MIN  # unused variable (poc_homography/homography/intrinsic_extrinsic.py:122) - class constant
_.compute_from_config  # unused method (poc_homography/homography/intrinsic_extrinsic.py:481)
_.project_point  # unused method (poc_homography/homography/intrinsic_extrinsic.py:603) - interface impl
_.project_points  # unused method (poc_homography/homography/intrinsic_extrinsic.py:626)
_.get_confidence  # unused method (poc_homography/homography/intrinsic_extrinsic.py:639)
_.project_point_static  # unused method (poc_homography/homography/intrinsic_extrinsic.py:721)
_.project_image_to_map  # unused method (poc_homography/homography/intrinsic_extrinsic.py:754)
_.result_to_homography_result  # unused method (poc_homography/homography/intrinsic_extrinsic.py:804)
_.load_lens_distortion_calibration  # unused method (poc_homography/homography/intrinsic_extrinsic.py:813)
_.merge_calibration_tables  # unused method (poc_homography/homography/intrinsic_extrinsic.py:839)
inverse_matrix  # unused variable (poc_homography/homography/map_points.py:46)
inverse_matrix  # unused variable (poc_homography/homography/map_points.py:71)
_._point_counter  # unused attribute (poc_homography/homography/map_points.py:99)
iteration  # unused variable (poc_homography/homography/map_points.py:336)
_.camera_to_map_batch  # unused method (poc_homography/homography/map_points.py:618)
_.map_to_camera_batch  # unused method (poc_homography/homography/map_points.py:652)
_.get_result  # unused method (poc_homography/homography/map_points.py:676)
_.get_homography_matrix  # unused method (poc_homography/homography/map_points.py:680)
_.get_inverse_matrix  # unused method (poc_homography/homography/map_points.py:684)
_.writeable  # unused attribute (poc_homography/homography/parameters.py:175)
_.writeable  # unused attribute (poc_homography/homography/parameters.py:186)
_.from_reference_dict  # unused method (poc_homography/homography/parameters.py:273)
_.to_reference_dict  # unused method (poc_homography/homography/parameters.py:338)
_.writeable  # unused attribute (poc_homography/homography/parameters.py:460)
_.writeable  # unused attribute (poc_homography/homography/parameters.py:471)
_.writeable  # unused attribute (poc_homography/homography/parameters.py:482)
_.map_dimensions  # unused property (poc_homography/homography/parameters.py:485)
_.last_ptz_state  # unused property (poc_homography/infrastructure/clients/hikvision_camera_controller.py:53)
_.move_relative  # unused method (poc_homography/infrastructure/clients/hikvision_camera_controller.py:119)
_.get_by_map  # unused method (poc_homography/infrastructure/repositories/base/mixin_repo_map_filter.py:15)
_.clear_cache  # unused method (poc_homography/infrastructure/repositories/base/repo_yaml.py:98)
_.get_by_map  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:129)
_.get_by_camera  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:152)
_.clear_cache  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:182)
_.save_annotations  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:233)
CameraAnnotatorConfig  # unused class (webapp/camera_annotator/apps.py:6)
default_auto_field  # unused variable (webapp/camera_annotator/apps.py:9)
verbose_name  # unused variable (webapp/camera_annotator/apps.py:11)
app_name  # unused variable (webapp/camera_annotator/urls.py:7)
urlpatterns  # unused variable (webapp/camera_annotator/urls.py:9)
CameraDiagnosticConfig  # unused class (webapp/camera_diagnostic/apps.py:6)
default_auto_field  # unused variable (webapp/camera_diagnostic/apps.py:9)
verbose_name  # unused variable (webapp/camera_diagnostic/apps.py:11)
_.get_overall_status  # unused method (webapp/camera_diagnostic/models.py:119)
GOOD  # unused variable (webapp/camera_diagnostic/models.py:248)
NEEDS_IMPROVEMENT  # unused variable (webapp/camera_diagnostic/models.py:249)
BAD  # unused variable (webapp/camera_diagnostic/models.py:250)
app_name  # unused variable (webapp/camera_diagnostic/urls.py:7)
urlpatterns  # unused variable (webapp/camera_diagnostic/urls.py:9)
CameraEvaluationConfig  # unused class (webapp/camera_evaluation/apps.py:6)
default_auto_field  # unused variable (webapp/camera_evaluation/apps.py:9)
verbose_name  # unused variable (webapp/camera_evaluation/apps.py:11)
app_name  # unused variable (webapp/camera_evaluation/urls.py:11)
urlpatterns  # unused variable (webapp/camera_evaluation/urls.py:13)
CameraLineAnnotatorConfig  # unused class (webapp/camera_line_annotator/apps.py:6)
default_auto_field  # unused variable (webapp/camera_line_annotator/apps.py:9)
app_name  # unused variable (webapp/camera_line_annotator/urls.py:7)
urlpatterns  # unused variable (webapp/camera_line_annotator/urls.py:9)
_.modified  # unused attribute (webapp/camera_line_annotator/views.py:98)
CameraSurveyConfig  # unused class (webapp/camera_survey/apps.py:6)
default_auto_field  # unused variable (webapp/camera_survey/apps.py:9)
verbose_name  # unused variable (webapp/camera_survey/apps.py:11)
camera_model  # unused variable (webapp/camera_survey/ptz.py:329)
app_name  # unused variable (webapp/camera_survey/urls.py:7)
urlpatterns  # unused variable (webapp/camera_survey/urls.py:9)
app_name  # unused variable (webapp/distortion_validator/urls.py:7)
urlpatterns  # unused variable (webapp/distortion_validator/urls.py:9)
GcpConfig  # unused class (webapp/gcp/apps.py:4)
default_auto_field  # unused variable (webapp/gcp/apps.py:5)
VALID_LAYERS  # unused variable (webapp/gcp/templatetags/satellite_layers.py:27)
satellite_layers_js  # unused function (webapp/gcp/templatetags/satellite_layers.py:39)
app_name  # unused variable (webapp/gcp/urls.py:11)
urlpatterns  # unused variable (webapp/gcp/urls.py:24)
default_app_config  # unused variable (webapp/homography_precision/__init__.py:3)
HomographyPrecisionConfig  # unused class (webapp/homography_precision/apps.py:6)
default_auto_field  # unused variable (webapp/homography_precision/apps.py:9)
verbose_name  # unused variable (webapp/homography_precision/apps.py:11)
app_name  # unused variable (webapp/homography_precision/urls.py:11)
urlpatterns  # unused variable (webapp/homography_precision/urls.py:46)
homography_source  # unused variable (webapp/homography_precision/views.py:795)
line_result  # unused variable (webapp/homography_precision/views.py:810)
application  # unused variable (webapp/homography_web/asgi.py:16)
SECRET_KEY  # unused variable (webapp/homography_web/settings.py:26)
ALLOWED_HOSTS  # unused variable (webapp/homography_web/settings.py:31)
INSTALLED_APPS  # unused variable (webapp/homography_web/settings.py:36)
MIDDLEWARE  # unused variable (webapp/homography_web/settings.py:56)
ROOT_URLCONF  # unused variable (webapp/homography_web/settings.py:66)
TEMPLATES  # unused variable (webapp/homography_web/settings.py:68)
WSGI_APPLICATION  # unused variable (webapp/homography_web/settings.py:83)
DATABASES  # unused variable (webapp/homography_web/settings.py:89)
AUTH_PASSWORD_VALIDATORS  # unused variable (webapp/homography_web/settings.py:100)
LANGUAGE_CODE  # unused variable (webapp/homography_web/settings.py:119)
TIME_ZONE  # unused variable (webapp/homography_web/settings.py:121)
USE_I18N  # unused variable (webapp/homography_web/settings.py:123)
USE_TZ  # unused variable (webapp/homography_web/settings.py:125)
STATIC_URL  # unused variable (webapp/homography_web/settings.py:131)
STATICFILES_DIRS  # unused variable (webapp/homography_web/settings.py:132)
DEFAULT_AUTO_FIELD  # unused variable (webapp/homography_web/settings.py:139)
urlpatterns  # unused variable (webapp/homography_web/urls.py:21)
application  # unused variable (webapp/homography_web/wsgi.py:16)
LensCalibrationConfig  # unused class (webapp/lens_calibration/apps.py:6)
default_auto_field  # unused variable (webapp/lens_calibration/apps.py:9)
verbose_name  # unused variable (webapp/lens_calibration/apps.py:11)
app_name  # unused variable (webapp/lens_calibration/urls.py:7)
urlpatterns  # unused variable (webapp/lens_calibration/urls.py:9)
LinePickerConfig  # unused class (webapp/line_picker/apps.py:9)
default_auto_field  # unused variable (webapp/line_picker/apps.py:12)
verbose_name  # unused variable (webapp/line_picker/apps.py:14)
_.ready  # unused method (webapp/line_picker/apps.py:16)
app_name  # unused variable (webapp/line_picker/urls.py:7)
urlpatterns  # unused variable (webapp/line_picker/urls.py:9)
default_app_config  # unused variable (webapp/point_picker/__init__.py:3)
PointPickerConfig  # unused class (webapp/point_picker/apps.py:9)
default_auto_field  # unused variable (webapp/point_picker/apps.py:12)
verbose_name  # unused variable (webapp/point_picker/apps.py:14)
_.ready  # unused method (webapp/point_picker/apps.py:16)
app_name  # unused variable (webapp/point_picker/urls.py:7)
urlpatterns  # unused variable (webapp/point_picker/urls.py:9)
DATA_DIR  # unused variable (webapp/homography_web/settings.py:19)
invalidate_cache  # unused function (webapp/homography_web/frame_utils.py) - public API
_.get_by_frame_id  # unused method (poc_homography/infrastructure/repositories/repo_yaml_annotation.py:15) - repository query API
_invalidate_line_registry_cache  # unused function (webapp/homography_precision/views.py:88) - cache management
get_annotation_repo  # unused function (webapp/homography_web/frame_utils.py:220) - public API
save_to_line_repo  # unused function (webapp/line_picker/state.py) - used by tests
list_line_map_ids  # unused function (webapp/line_picker/state.py) - used by tests
TenantIdMiddleware  # unused class (webapp/homography_web/middleware.py) - Django MIDDLEWARE setting
_.process_exception  # unused method (webapp/homography_web/middleware.py) - Django middleware hook
_.get_coefficients  # unused method (poc_homography/calibration/lens_distortion/calibration_table.py:232) - public API
_.get_entry  # unused method (poc_homography/calibration/lens_distortion/calibration_table.py:377) - public API
straightness_rmse  # unused function (poc_homography/calibration/lens_distortion/distortion_solver.py:474) - utility
get_rtsp_url  # unused function (poc_homography/camera_config.py:525) - used by webapp views
_.focal_length_at_zoom  # unused method (poc_homography/domain/enums/camera_spec.py:101) - domain API
_.delete  # unused method (poc_homography/domain/repositories/repo.py:35) - repository protocol
_.to_list  # unused method (poc_homography/domain/vo/geotiff.py:38) - VO API
_.pixel_to_geo  # unused method (poc_homography/domain/vo/geotiff.py:85) - VO API
_.area  # unused property (poc_homography/domain/vo/image_dimensions.py:46) - VO API
_.center_x  # unused property (poc_homography/domain/vo/image_dimensions.py:51) - VO API
_.center_y  # unused property (poc_homography/domain/vo/image_dimensions.py:56) - VO API
_.has_distortion  # unused property (poc_homography/domain/vo/lens_distortion.py:40) - VO API
_.inverse  # unused method (poc_homography/domain/vo/matrix3x3.py:128) - VO API
_.to_list  # unused method (poc_homography/domain/vo/matrix3x3.py:171) - VO API
_.to_orientation  # unused method (poc_homography/domain/vo/ptz_state.py:30) - VO API
_.inverse  # unused method (poc_homography/homography/interface.py:98) - interface API
_.transform  # unused method (poc_homography/homography/interface.py:134) - interface API
num_gcps  # unused variable (poc_homography/homography/map_points.py:47) - NamedTuple field
num_lines  # unused variable (poc_homography/homography/map_points.py:72) - NamedTuple field
mean_perp_error  # unused variable (poc_homography/homography/map_points.py:75) - NamedTuple field
max_perp_error  # unused variable (poc_homography/homography/map_points.py:76) - NamedTuple field
_.compute_from_gcps  # unused method (poc_homography/homography/map_points.py:118) - core algorithm
_.compute_from_lines  # unused method (poc_homography/homography/map_points.py:238) - core algorithm
_.camera_to_map  # unused method (poc_homography/homography/map_points.py:575) - core algorithm
_.map_to_camera  # unused method (poc_homography/homography/map_points.py:601) - core algorithm
_.get_by_tenant  # unused method (poc_homography/infrastructure/repositories/base/mixin_repo_tenant_filter.py:15) - mixin API
_.delete  # unused method (poc_homography/infrastructure/repositories/base/repo_yaml.py:84) - repository CRUD
_.image_dir_for  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:35) - repo API
_.delete  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:99) - repo CRUD
_.get_image_path  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:201) - repo API
_.get_annotations  # unused method (poc_homography/infrastructure/repositories/repo_yaml_captured_frame.py:226) - repo API
_.delete  # unused method (poc_homography/infrastructure/repositories/repo_yaml_diagnostic_session.py:101) - repo CRUD
_.get_by_map_id  # unused method (poc_homography/infrastructure/repositories/repo_yaml_line.py:15) - repo query
_.delete  # unused method (poc_homography/infrastructure/repositories/repo_yaml_stress_test_session.py:80) - repo CRUD
_.delete  # unused method (poc_homography/infrastructure/repositories/repo_yaml_survey_session.py:86) - repo CRUD
_.get_session_dir  # unused method (poc_homography/infrastructure/repositories/repo_yaml_survey_session.py:107) - repo API
from_gcp_repo  # unused function (poc_homography/map_points/gcp_registry.py:172) - legacy bridge
save_to_gcp_repo  # unused function (poc_homography/map_points/gcp_registry.py:200) - legacy bridge
list_map_ids  # unused function (poc_homography/map_points/gcp_registry.py:227) - legacy bridge
error_px  # unused variable (poc_homography/validation/camera_model.py:65) - NamedTuple field

# -- PostgreSQL _pg adapter functions (called from api/ routers, not scanned by vulture) --
sync_to_ddd_repo_pg  # unused function (poc_homography/calibration/lens_distortion/ddd_sync.py) - used by api/routers/lens_calibration.py
get_tenants_pg  # unused function (poc_homography/camera_config.py) - used by api/routers
get_tenant_by_id_pg  # unused function (poc_homography/camera_config.py) - used by api/routers
get_tenant_by_name_pg  # unused function (poc_homography/camera_config.py) - used by api/routers
save_to_gcp_repo_pg  # unused function (poc_homography/map_points/gcp_registry.py) - used by api/routers/point_picker.py
list_map_ids_pg  # unused function (poc_homography/map_points/gcp_registry.py) - used by api/routers/point_picker.py
save_line_to_repo_pg  # unused function (webapp/line_picker/state.py) - used by api/routers/line_picker.py
delete_line_from_repo_pg  # unused function (webapp/line_picker/state.py) - used by api/routers/line_picker.py
save_gcp_to_repo_pg  # unused function (webapp/point_picker/state.py) - used by api/routers/point_picker.py
delete_gcp_from_repo_pg  # unused function (webapp/point_picker/state.py) - used by api/routers/point_picker.py

# -- PostgreSQL session repository base class (abstract hooks called by base CRUD) --
_.get_session_dir  # unused method (infrastructure/repositories/repo_postgres_survey_session.py) - YAML compat API

# -- SQLAlchemy ORM relationship attributes (used by SA for joins/back_populates) --
_.map  # unused attribute - SA relationship (infrastructure/models/camera_config.py)
_.camera_calibration  # unused attribute - SA relationship (infrastructure/models/camera_config.py)
_.lens_calibration_table  # unused attribute - SA relationship (infrastructure/models/camera_config.py)
_.map  # unused attribute - SA relationship (infrastructure/models/captured_frame.py)
_.map  # unused attribute - SA relationship (infrastructure/models/ground_control_point.py)
_.map  # unused attribute - SA relationship (infrastructure/models/line.py)
_.camera_configs  # unused attribute - SA relationship (infrastructure/models/map.py)
_.captured_frames  # unused attribute - SA relationship (infrastructure/models/map.py)
_.ground_control_points  # unused attribute - SA relationship (infrastructure/models/map.py)
_.camera_configs  # unused attribute - SA relationship (infrastructure/models/tenant.py)
_.users  # unused attribute - SA relationship (infrastructure/models/tenant.py)

# -- SQLAlchemy ORM column attributes (used by SA for mapping) --
_.hashed_password  # unused attribute - SA column (infrastructure/models/user.py)
_.is_active  # unused attribute - SA column (infrastructure/models/user.py)
_.updated_at  # unused attribute - SA column (infrastructure/models/user.py)


# -- Hikvision ISAPI adapter Phase A public surface (consumed by Phase C client) --
# Endpoint path builders (poc_homography/infrastructure/clients/hikvision/isapi_endpoints.py)
ptz_capabilities  # unused function - ISAPI endpoint builder
ptz_absolute_ex_capabilities  # unused function - ISAPI endpoint builder
ptz_absolute  # unused function - ISAPI endpoint builder
ptz_relative  # unused function - ISAPI endpoint builder
ptz_continuous  # unused function - ISAPI endpoint builder
ptz_momentary  # unused function - ISAPI endpoint builder
ptz_position3d  # unused function - ISAPI endpoint builder
ptz_presets  # unused function - ISAPI endpoint builder
ptz_preset_goto  # unused function - ISAPI endpoint builder
ptz_home_goto  # unused function - ISAPI endpoint builder
system_status  # unused function - ISAPI endpoint builder
streaming_channel  # unused function - ISAPI endpoint builder
streaming_picture  # unused function - ISAPI endpoint builder
image_focus_configuration  # unused function - ISAPI endpoint builder
image_iris  # unused function - ISAPI endpoint builder
image_exposure  # unused function - ISAPI endpoint builder
image_white_balance  # unused function - ISAPI endpoint builder
image_capabilities  # unused function - ISAPI endpoint builder
# Unit conversions (poc_homography/infrastructure/clients/hikvision/isapi_units.py)
raw_to_degrees  # unused function - PTZ unit conversion
degrees_to_raw  # unused function - PTZ unit conversion
raw_to_zoom  # unused function - PTZ unit conversion
zoom_to_raw  # unused function - PTZ unit conversion
# Transport public API (poc_homography/infrastructure/clients/hikvision/isapi_transport.py)
_.get_xml  # unused method - IsapiTransport public API
_.put_xml  # unused method - IsapiTransport public API
_.get_bytes  # unused method - IsapiTransport public API
_.auth  # unused attribute - requests.Session.auth assignment

# -- Hikvision ISAPI adapter Phase C public surface --
# CameraDevice protocol (poc_homography/domain/protocols/camera_device.py) +
# HikvisionISAPIClient adapter (.../hikvision/isapi_client.py); consumed by
# Phase D tests and Phase E callers.
_.from_config  # unused method - HikvisionISAPIClient factory
_.get_health  # unused method - CameraDevice / adapter API
_.get_stream_profiles  # unused method - CameraDevice / adapter API
_.stop  # unused method - CameraDevice / adapter API
_.goto_preset  # unused method - CameraDevice / adapter API
_.position3d  # unused method - CameraDevice / adapter API
_.get_optics  # unused method - CameraDevice / adapter API
_.set_focus  # unused method - CameraDevice / adapter API
_.set_iris  # unused method - CameraDevice / adapter API
_.set_exposure  # unused method - CameraDevice / adapter API
_.list_presets  # unused method - CameraDevice / adapter API
_.capture_snapshot  # unused method - CameraDevice / adapter API
_assert_camera_device  # unused function - TYPE_CHECKING structural conformance check
_.discover_endpoints  # adapter endpoint-probe used by ptz_discovery_and_control/hikvision/discover.py CLI (outside vulture scan paths) [#256]
_.get_ptz_state  # HikvisionPTZCamera pass-through consumed by api/routers/camera_evaluation.py (outside vulture scan paths) [#256]

# -- Survey dataset schema (#258); consumed by capture/planner/phase issues C2-C5 --
# SurveyPhase enum members (poc_homography/domain/enums/survey_phase.py)
CAMERA_INVENTORY  # unused enum member - SurveyPhase value used by planner/phases
PTZ_CHARACTERIZATION  # unused enum member - SurveyPhase value used by planner/phases
ZOOM_CHARACTERIZATION  # unused enum member - SurveyPhase value used by planner/phases
DENSE_NADIR  # unused enum member - SurveyPhase value used by planner/phases
MAIN_SURVEY  # unused enum member - SurveyPhase value used by planner/phases
CROSS_ZOOM  # unused enum member - SurveyPhase value used by planner/phases
REPEATABILITY  # unused enum member - SurveyPhase value used by planner/phases
STATIC_JITTER  # unused enum member - SurveyPhase value used by planner/phases
VALIDATION  # unused enum member - SurveyPhase value used by planner/phases
# VideoBurstRecord + RepoYamlSurveyRun grouping query surface
_.frame_by_index  # unused method - VideoBurstRecord offline-processing accessor (#258)
_.get_frames_by_run  # unused method - RepoYamlSurveyRun grouping query (#258)
_.get_frames_by_phase  # unused method - RepoYamlSurveyRun grouping query (#258)
_.get_frames_by_camera  # unused method - RepoYamlSurveyRun grouping query (#258)
_.get_frames_by_zoom_range  # unused method - RepoYamlSurveyRun grouping query (#258)
_.get_frames_by_burst  # unused method - RepoYamlSurveyRun grouping query (#258)
SurveyCaptureEngine  # C2 capture engine; consumed by C3/C4 run lifecycle (poc_homography/infrastructure/survey/capture_engine.py)
_.capture_snapshot_burst  # public capture path; called by C3/C4 (poc_homography/infrastructure/survey/capture_engine.py)
_.capture_video_burst  # public capture path; called by C3/C4 (poc_homography/infrastructure/survey/capture_engine.py)
# Multi-phase survey planner public API (#260); consumed by C3/C4 run lifecycle
PAUSED  # SurveyRunStatus member for RUNNING->PAUSED->RUNNING resume (poc_homography/domain/enums/survey_run_status.py)
_.remaining_poses  # PlannedSurveyRun resume iterator (poc_homography/survey/planner/run.py)
_.advance  # PlannedSurveyRun cursor advance (poc_homography/survey/planner/run.py)
_.with_status  # PlannedSurveyRun status transition helper (poc_homography/survey/planner/run.py)
_.header  # PlannedSurveyRun -> C1 SurveyRun persistence bridge (poc_homography/survey/planner/run.py)

# -- Survey phases C4 (#261); consumed by C5 (CLI/endpoints) + tests (outside vulture scan paths) --
YamlPhaseSink  # YAML PhaseSink adapter; wired by C5 + survey-phase tests (poc_homography/infrastructure/survey/yaml_phase_sink.py)

# -- Survey operator surface C5 (#262); typer commands + plan-config repo contract --
run_command  # typer survey command (poc_homography/cli/survey.py)
status_command  # typer survey command (poc_homography/cli/survey.py)
abort_command  # typer survey command (poc_homography/cli/survey.py)
list_command  # typer survey command (poc_homography/cli/survey.py)
browse_command  # typer survey command (poc_homography/cli/survey.py)
_.save_plan_config  # SurveyRunRepository plan-config contract; consumed by tests + future C3 (poc_homography/domain/protocols/survey_run_repository.py)
_.load_plan_config  # SurveyRunRepository plan-config contract; consumed by tests + future C3 (poc_homography/domain/protocols/survey_run_repository.py)
_.detail  # ValidationOutcome human-readable note; injectable-hook contract (poc_homography/horizon/validation.py)
_.save_pose_catalog  # SurveyRunRepository pose-catalog contract (#276); consumed by tests + future C2/C4 (poc_homography/domain/protocols/survey_run_repository.py)
_.get_runs_by_camera_and_pose  # SurveyRunRepository multi-visit grouping query (#276); consumed by tests (poc_homography/domain/protocols/survey_run_repository.py)

# -- Phase-0 horizon calibration C3 (#275); step-1 entry point consumed by C5/CLI + tests --
calibrate_horizon_envelope  # Phase-0 calibration step; invoked by operator/CLI + tests (poc_homography/survey/calibration.py)

# -- Clean-plate capture domain (#276); PoseCatalog public API consumed by C2/C4 + tests --
_.with_pose  # PoseCatalog immutable pose-record builder (poc_homography/domain/entities/survey/pose_catalog.py)
_.from_poses  # PoseCatalog deterministic builder from a pose sequence (poc_homography/domain/entities/survey/pose_catalog.py)
_.from_plan_config  # SurveyPlanConfig->SurveyPlan burst-count bridge; consumed by C5 + tests (poc_homography/survey/phases/runner.py)

# -- Clean-plate reconstruction (#277); GroundRaster public coord API + frame metadata consumed by downstream loader/CLI + tests --
_.world_to_cell  # GroundRaster public world->cell mapping; exported API (poc_homography/cleanplate/raster.py)
_.cell_to_world  # GroundRaster public cell->world mapping; exported API (poc_homography/cleanplate/raster.py)
time_bucket  # CleanPlateFrame exposure metadata field; consumed by photometric/loader (poc_homography/cleanplate/reconstruct.py)
_.from_survey_run  # CleanPlateDataset survey-run loader entry point; consumed by tests + future CLI (poc_homography/cleanplate/dataset.py)
_.frames_for  # CleanPlateDataset per-group frame materialiser; consumed by tests + reconstruct_clean_plate caller (poc_homography/cleanplate/dataset.py)

# -- Clean-plate CLI (#277); typer commands invoked via `hom cleanplate ...` --
reconstruct_command  # typer cleanplate command (poc_homography/cli/cleanplate.py)
synth_command  # typer cleanplate command (poc_homography/cli/cleanplate.py)

# -- Clean-plate frames table (#283); MinIO location columns consumed by PostgresPhaseSink + gallery (#284/#285) --
_.minio_bucket  # CleanPlateFrameModel MinIO bucket column (poc_homography/infrastructure/models/clean_plate_frame.py)
_.minio_object_key  # CleanPlateFrameModel MinIO object-key column (poc_homography/infrastructure/models/clean_plate_frame.py)
_.checksum_sha256  # CleanPlateFrameModel image content-hash column (poc_homography/infrastructure/models/clean_plate_frame.py)

# -- MinIO frame store (#284); entry points consumed by capture wiring + gallery (#285) --
_.from_env  # MinioFrameStore env constructor; used by the maglor capture script (poc_homography/infrastructure/clients/minio_frame_store.py)
_.presign_get  # MinioFrameStore presigned-GET URL minting; consumed by the gallery API (#285) (poc_homography/infrastructure/clients/minio_frame_store.py)
