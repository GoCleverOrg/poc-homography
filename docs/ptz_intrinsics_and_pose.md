Here is the breakdown for the Hikvision DS-2DF8425IX-AELW(T5).

### 1\. Computing Intrinsics ($K$)

The Intrinsic matrix maps 3D camera coordinates to 2D pixel coordinates.
$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

You need three specific values from the datasheet:

1.  **Sensor Width ($S_w$):** The spec is **1/1.8"** Progressive Scan CMOS. The standard width for this format is **7.18 mm**.
2.  **Image Width ($W_{px}$):** Max resolution is **2560 px** (width).
3.  **Focal Length ($F_{mm}$):** Range is **5.9 mm to 147.5 mm** (25x Zoom).

**The Algorithm:**

1.  **Convert Zoom to mm:** The camera is linear.
    $$F_{current\_mm} = 5.9 \times \text{current\_zoom\_factor}$$
2.  **Convert mm to pixels ($f_x, f_y$):**
    $$f_x = F_{current\_mm} \times \frac{W_{px}}{S_w}$$
    *(Note: $f_y$ is usually equal to $f_x$ for square pixels).*
3.  **Principal Point ($c_x, c_y$):** Assume the center of the image.
    $$c_x = W_{px} / 2, \quad c_y = H_{px} / 2$$

**Python Snippet:**

```python
import numpy as np

def get_intrinsics(zoom_factor, W_px=2560, H_px=1440, sensor_width_mm=7.18):
    # 1. Calculate focal length in mm (Linear mapping based on datasheet: 1x=5.9mm)
    f_mm = 5.9 * zoom_factor
    
    # 2. Convert focal length to pixels
    f_px = f_mm * (W_px / sensor_width_mm)
    
    # 3. Construct K
    cx, cy = W_px / 2.0, H_px / 2.0
    K = np.array([
        [f_px, 0,    cx],
        [0,    f_px, cy],
        [0,    0,    1 ]
    ])
    return K
```

-----

### 2\. Computing Camera Position & Orientation

You have $K$ and a way to relate pixels to the floor. Here is how to get the pose $(R, t)$.

#### Method A: From Homography Matrix (The Analytic Way)

If you have the Homography $H$ (Image $\to$ Floor), you can decompose it.
*Concept:* $H$ is essentially $K \times [r_1 r_2 t]$ without the Z-column (since $Z=0$).

1.  **Remove Intrinsics:** Calculate matrix $A = K^{-1} H$.
      * Let $A = [h_1, h_2, h_3]$ (columns).
2.  **Normalize:** The matrix scale is arbitrary. The true scale $\lambda$ is found because rotation columns must be unit length.
      * $\lambda = 1 / \|h_1\|$
3.  **Recover Extrinsics:**
      * $r_1 = \lambda h_1$
      * $r_2 = \lambda h_2$
      * $r_3 = r_1 \times r_2$ (Orthogonal vector)
      * $t = \lambda h_3$
      * $R = [r_1, r_2, r_3]$
4.  **Compute Position:**
      * **Position:** $P = -R^T t$
      * **Orientation Vector:** The 3rd row of $R$ (or $R[:, 2]$ depending on convention) is the look-at vector.

#### Method B: From 4+ Points (The Developer Way)

This is usually more numerically stable than manually decomposing H. You use a PnP (Perspective-n-Point) solver.

  * **Inputs:**
      * `objectPoints`: Array of $(X, Y, 0)$ coordinates on the floor.
      * `imagePoints`: Array of $(u, v)$ pixel coordinates.
      * `K`: Your intrinsic matrix.

**Python Snippet:**

```python
import cv2

# Assume floor_pts (3D) and pixel_pts (2D) are defined
success, rvec, tvec = cv2.solvePnP(floor_pts, pixel_pts, K, distCoeffs=None)

# Convert rotation vector to matrix
R, _ = cv2.Rodrigues(rvec)

# 1. Camera Position (Height is the Z component)
camera_position = -R.T @ tvec

# 2. Visual Orientation Vector (Look-at direction in World Frame)
# The camera looks down its own Z-axis (0,0,1). 
# We rotate that vector into the world frame.
orientation_vector = R.T @ np.array([0, 0, 1]) 
```