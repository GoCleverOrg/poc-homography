# Decision Log: Intrinsic-Extrinsic Homography Propagation

## 1. Mathematical Approach: Model-Based vs. Point-Based
**Decision:** Use a **Model-Based (Physical)** approach to derive Homography.
**Context:** There are two ways to find a Homography matrix ($H$):
1.  **Point-Correspondence:** Click 4 points on image, 4 on map, solve linear equations.
2.  **Physical Model:** Construct $H$ from Intrinsic ($K$) and Extrinsic ($R, t$) matrices.

**Reasoning:**
* The user requested "Intrinsic Extrinsic Propagation".
* **Propagation capability:** If the camera moves (e.g., a PTZ camera), we can simply update the rotation angles (Extrinsics) in the code, and the projection remains accurate without re-labeling points.
* **Physics-aware:** This allows us to filter objects based on real-world size constraints later.

## 2. Coordinate System Definition
**Decision:**
* **World Frame:** Origin is on the ground directly below the camera. $Z=0$ is the ground plane.
* **Camera Frame:** Standard OpenCV convention ($Z$ forward, $X$ right, $Y$ down).

## 3. The Projection Chain
We implement the standard Pinhole Camera Model:
$$s \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \cdot [R | t] \cdot \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Where:
* **$K$ (Intrinsics):** Contains Focal Length ($f_x, f_y$) and Optical Center ($c_x, c_y$).
* **$[R|t]$ (Extrinsics):** Defines camera height and tilt (pitch/yaw/roll).
* **Homography ($H$):** Since we map the ground where $Z_w=0$, we remove the 3rd column of the rotation matrix to form a $3 \times 3$ Homography matrix.

## 4. Output Scaling
**Decision:** Implement a `pixels_per_meter` scaling factor.
**Reasoning:** The Homography projects 3D world meters to 2D map pixels. We need a scalar to convert the resulting "World Meters" into "Side Panel Pixels" so the dots fit nicely on the side panel image.


# Decisions for Refactoring to Support RTSP Stream

The goal of the refactoring was to integrate live RTSP stream processing into the `VideoAnnotator` class while retaining the original file-based video processing (`process_video`).

## 1. Class Initialization (`__init__`)

* **Decision:** Made `video_path` parameter optional (`Optional[str] = None`).
* **Reasoning:** To allow the class to be instantiated without a file path when operating in stream mode. The presence of `video_path` now dictates whether the object will attempt file loading or stream loading.

## 2. Stream Loading Method (`load_stream`)

* **Decision:** Created a new method, `load_stream(rtsp_url: str)`, to handle initialization specific to RTSP.
* **Reasoning:**
    * RTSP requires the `cv2.CAP_FFMPEG` flag for reliable network connection.
    * Unlike files, streams have no known `CAP_PROP_FRAME_COUNT` (set to `-1`).
    * The resolution (`frame_width`, `frame_height`) must be derived by reading the **first successful frame** after connection, not by querying properties on the un-opened capture object.
    * The `fps` for recording must be set to a reasonable default (e.g., `30`) as the actual stream speed is often variable and unreliable to query.

## 3. Stream Processing Method (`process_stream`)

* **Decision:** Created a dedicated main loop method, `process_stream(rtsp_url, box_generator)`.
* **Reasoning:** The logic for stream processing differs fundamentally from file processing:
    * **Termination:** Uses `cv2.imshow` and checks `cv2.waitKey(1) & 0xFF == ord('q')` for user exit, replacing the "end-of-file" check (`if not ret: break`).
    * **Reconnection:** Includes basic error handling and an attempt to reconnect (`time.sleep(5)`) if a frame fails to read (`if not ret:`), common for unreliable network streams.
    * **Progress:** Removed file-specific progress indicator and replaced it with a simple frame count and final average FPS calculation.

## 4. **Display Frame Preview (New Requirement)**

* **Decision:** Ensured `cv2.imshow('Live Stream (Press Q to Quit and Save)', expanded)` is explicitly called within the `process_stream` loop.
* **Reasoning:** This fulfills the requirement to see a small preview window of the frames being processed, including the live annotation and the expanded top-down view. The `expanded` frame is used for display as it provides the most comprehensive visual feedback, showing both the camera view and the homography/map view simultaneously.

## 5. Helper Function Adjustments

* **`_generate_example_boxes` and `_generate_example_tracked_points`:** No changes were necessary inside these methods, as they already rely on `frame_width`, `frame_height`, and `frame_number`, which are correctly initialized and incremented in the new `process_stream` loop.
* **`setup_writers`:** No changes required, as the necessary properties (`fps`, `frame_width`, `frame_height`) are guaranteed to be set by the preceding `load_stream` call.

Decisions for camera_geometry.py ClassDecision: Created a new, dedicated CameraGeometry class to encapsulate all 3D-to-2D projection logic.Reasoning: Follows the Single Responsibility Principle, isolating geometry and linear algebra from video processing/annotation. This improves code structure and maintainability.Decision: Implemented _get_rotation_matrix() to compute the rotation $R$ based on known pan (Yaw) and tilt (Pitch) angles.Reasoning: To correctly derive the homography for a real-world static camera, the camera's orientation is essential. This method uses a standard Euler angle composition (Z then X rotation) to create the $3 \times 3$ rotation matrix $R$.Decision: Implemented _calculate_ground_homography() to compute the Homography matrix $H$ using the formula $H = K [\mathbf{r}_1, \mathbf{r}_2, \mathbf{t}]$.The translation vector $\mathbf{t}$ is calculated as $t = -R \mathbf{w}_{pos}$.The resulting $H$ is stored in self.H and its inverse, $\mathbf{H}^{-1}$, is stored in self.H_inv.Reasoning: This is the correct method for perspective projection of a ground plane defined as the world $Z=0$ plane. Storing $\mathbf{H}^{-1}$ is necessary because the core task is mapping Image coordinates to World coordinates ($\mathbf{H}^{-1}$), not the other way around.Decision: Updated project_image_to_map() to use the calculated $\mathbf{H}^{-1}$ matrix.Reasoning: The function now performs the full projection pipeline: Image Pixels $\rightarrow$ World Meters $\rightarrow$ Map Pixels, utilizing the complex geometry setup. A placeholder scale of $100$ pixels per meter (PPM) was introduced for mapping the world coordinates onto the side panel.

Decisions for Geometry Setup and CoordinatesDecision: Introduced a new utility class, CoordinatesConverter, with a static method dms_to_dd() to parse geographic coordinates (DMS format).Reasoning: Isolates the coordinate system conversion logic, making it reusable and separating it from core camera geometry.Decision: Added get_intrinsics() as a static method to the CameraGeometry class.Reasoning: K is derived solely from camera hardware specs and zoom level, making it independent of any specific CameraGeometry instance state.Decision: In usage_example_stream(), the geographic coordinates are converted to Decimal Degrees and then arbitrarily mapped to local $X, Y$ meter offsets from a fictional origin, with $Z$ being the given $5\text{m}$ height.Reasoning: To satisfy the requirement of using $\mathbf{w}_{pos}$ in the homography calculation without introducing a complex geographic library (like GDAL or PyProj) for accurate UTM projection, a simplified local meter-based coordinate system is assumed for the homography demonstration.Decision: Updated annotate_stream_with_args to:Call annotator.load_stream() before setting geometry to ensure frame_width and frame_height are known.Calculate $K$ using annotator.geo.get_intrinsics().Create the $\mathbf{w}_{pos}$ vector (X, Y, Z meters).Call annotator.geo.set_camera_parameters() with the calculated $K$, $\mathbf{w}_{pos}$, and the new pan/tilt parameters.Reasoning: This establishes the entire geometric pipeline immediately after the stream size is determined, ensuring that the self.geo.H matrix is ready before the main loop begins.