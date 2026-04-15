# Camera Calibration Guide

Camera calibration is **critical** for accurate 3D measurements in this ArUco gap measurement system.

---

## 🎯 Why Calibration Matters

The system converts 2D pixel coordinates from the camera into real-world 3D coordinates. Without proper calibration:

- Distance measurements will have significant error
- Tilt angle calculations will be inaccurate
- The geometric intersection mathematics will be unreliable

---

## 🛠️ Prerequisites

**Before running any calibration scripts, ensure you have followed the [Prerequisites section in the main README](../README.md#prerequisites)** (virtual environment setup and system dependencies).

## 📋 Calibration Process

### 1. Prepare Chessboard Pattern

The calibration scripts expect a chessboard with these parameters (defined in `calibrate.py`):

- **Corners**: 10 × 7 (internal corners)
- **Square Size**: 15.0 mm

You can:
- Print a chessboard pattern (ensure squares are exactly 15mm)
- Use a pre-made calibration board
- Generate one using OpenCV tools

### 2. Capture Calibration Images

```bash
cd calibration
python capture_images.py
```

This utility helps you capture properly angled images of the chessboard. Best practices:

- **15-30 images** from different angles and distances
- Vary the orientation (tilt the board in all directions)
- Cover the full field of view
- Ensure good lighting with high contrast
- Keep the board completely still during capture
- Include extreme angles (the algorithm needs to see distortion)

### 3. Run Calibration

```bash
python calibrate.py
```

This script will:
1. Process all images in the `board_images/` directory
2. Detect chessboard corners in each image
3. Calculate camera intrinsic and distortion parameters
4. Save the results as `camera_params.npz`

**Expected Output:**
- Reprojection error (should be < 1.0 pixel for good calibration)
- Camera matrix and distortion coefficients
- Confirmation of successful calibration

---

## 📁 Output Files

The calibration produces:

- **`camera_params.npz`**: Main calibration file (preferred)
- **`camera_params_2.npz`**: Secondary calibration (for "Right" fixed side)

These files contain:
- `camera_matrix` / `mtx`: Intrinsic camera parameters
- `dist_coeff` / `dist`: Lens distortion coefficients

---

## 🔧 How the System Uses Calibration

The gap measurement engine (`modbus_communication/gap_engine.py` and `formulated/v17/camera.py`) automatically:

1. Looks for calibration files in this order:
   - `../calibration/camera_params.npz`
   - `./camera_params.npz`
   - Falls back to generic pinhole model

2. Selects between `camera_params.npz` and `camera_params_2.npz` based on the `fixed_side` configuration:
   - `"Left"` → `camera_params.npz`
   - `"Right"` → `camera_params_2.npz`

3. Uses the calibration to undistort images and convert 2D marker corners to 3D world coordinates.

---

## ⚙️ Configuration

In `modbus_communication/config.json`:

```json
{
  "fixed_side": "Left"
}
```

Change to `"Right"` if using the secondary calibration.

---

## 🔍 Troubleshooting Calibration

**Common Issues:**

**"Module not found" or "Command not found":**
- Ensure you have activated the virtual environment (see [Prerequisites](../README.md#prerequisites))

**"No chessboard found" / Low success rate:**
- Improve lighting contrast
- Ensure squares are exactly 15mm
- Check that the board isn't warped
- Try different angles

**High reprojection error (>2.0 pixels):**
- Need more images (aim for 20+)
- More variety in angles and positions
- Better image quality (less blur, better focus)

**System reports poor accuracy:**
- Verify the physical marker size in config matches reality
- Recalibrate if camera was moved or lens changed
- Check that `fixed_side` setting matches your calibration files

**File not found errors:**
- Ensure calibration files are in the `calibration/` folder OR root
- The system looks for both `camera_params.npz` and `camera_params_2.npz`

---

## 📈 Best Practices

1. **Calibrate in the actual environment** (same lighting, same mounting)
2. **Recalibrate if the camera is moved** or lens is changed
3. **Keep the calibration board clean** and flat
4. **Use the same resolution** for calibration and measurement (1280x720)
5. **Store multiple calibrations** for different setups

---

## 📊 Validation

After calibration, you can validate accuracy by:

1. Running the system with known gap distances
2. Checking that tilt angles read near zero when board is perpendicular
3. Comparing measurements against a physical gauge or caliper
4. Monitoring the reprojection error from the calibration process

---

See the main [`../README.md`](../README.md) for complete project documentation and the [`modbus_communication/README.md`](../modbus_communication/README.md) for running the final version.

**Good calibration = Sub-millimeter measurement accuracy.**
