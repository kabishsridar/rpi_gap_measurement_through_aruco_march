# Precision Dual-Pair ArUco Measurement Dashboard

A high-precision 3D measurement system designed for Raspberry Pi. This application uses two pairs of ArUco markers to calculate the gap distance between objects with sub-millimeter accuracy using a localized 3D geometric intersection method.

## 🚀 Features

- **Dual-Pair Support**: Simultaneously monitors a "Top Pair" and "Bottom Pair" of markers.
- **Vertical Isolation**: Advanced coordinate pairing ensures top markers only pair with top markers and bottom markers only pair with bottom markers.
- **Dynamic Settings**: Real-time adjustment of physical marker sizes for both pairs independently via the GUI.
- **Intelligent Math Fallback**: 
  - **Normal Mode**: Uses "Laser Hit" ray-casting intersection for maximum precision.
  - **High-Rotation Mode**: Automatically switches to edge-midpoint formulas if markers tilt beyond 20° to maintain stability.
- **Visual Safety Alerts**: Red-box warnings and yellow text appear instantly if marker tilt exceeds 5°, helping operators correct alignment.
- **Telemetry Logging**: Full 3D coordinate and angle data logging via the built-in `log.py` module.

## 🛠 Prerequisites

- **Hardware**: Raspberry Pi with a compatible Camera Module.
- **Calibration**: A `camera_params.npz` file must be present in the root directory for accurate 3D translation. If missing, the app uses generic pinhole model fallback.

## 📦 Installation

1. Clone the repository to your Raspberry Pi.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On Raspberry Pi, you may also need `sudo apt-get install python3-tk` for the GUI.*

## 🚦 How to Use

1. Run the dashboard:
   ```bash
   python v9_angle_correction.py
   ```
2. **Settings Tab**: Go to the "Configuration" tab to set the exact physical size (mm) of your ArUco markers.
3. **Live Monitor**: Watch the video feed. Measurement lines will appear in **Orange (Top)** and **Purple (Bottom)**.
4. **Telemetry**: Switch to the "Dual Telemetry" tab to view raw 3D positions, roll, and tilt for all four markers simultaneously.

## 📐 The Formula

The system solves for the intersection factor **k** between a directional ray from the source marker and the target edge of the secondary marker:
`k = [(v·w)(u·v) - (v·v)(u·w)] / [(v·v)(w·w) - (v·w)²]`
