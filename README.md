# RPi ArUco Gap Measurement System

**Industrial Precision Gap Measurement using Position-Based ArUco Markers (LT1, LT2, LB1, LB2, RT1, RT2, RB1, RB2) with Modbus PLC Integration**

A high-precision 3D measurement system designed for Raspberry Pi that uses two pairs of ArUco markers (Top and Bottom) to calculate gap distances with sub-millimeter accuracy. The final production version integrates with industrial PLCs via Modbus TCP.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture & Evolution](#architecture--evolution)
- [Prerequisites](#prerequisites)
- [Final Version (Recommended)](#final-version-recommended)
- [Project Structure](#project-structure)
- [Version History](#version-history)
- [Technical Details](#technical-details)
- [Calibration](#calibration)

---

## 🎯 Project Overview

This system measures the **gap between two objects** using computer vision:

- **Eight ArUco markers** positioned by location: **LT1, LT2, LB1, LB2** (Left side) and **RT1, RT2, RB1, RB2** (Right side)
- **L1-L2 pairs** (Left side: LT + LB markers) and **R1-R2 pairs** (Right side: RT + RB markers) enable side-specific gap measurements
- Uses **stereo-like 3D reconstruction** with a single camera + calibration
- Calculates the **intersection point** between marker planes using advanced geometric mathematics
- Provides real-time measurements with tilt detection and safety alerts
- **Industrial integration** via Modbus TCP to PLC systems

**Key Capabilities:**
- **Position-based marker naming**: LT1, LT2, LB1, LB2 (Left side), RT1, RT2, RB1, RB2 (Right side)
- **Side-based pairing**: "L1-L2 pair" (Left side markers) and "R1-R2 pair" (Right side markers) instead of just Top/Bottom
- Sub-millimeter precision gap measurement
- Real-time tilt/rotation monitoring (Roll, Pitch, Yaw)
- Automatic fallback between mathematical methods based on marker orientation
- Comprehensive logging (CSV + SQLite)
- Web-based industrial dashboard
- PLC communication (Modbus TCP)

---

## 🏗️ Architecture & Evolution

The project evolved through **17+ iterations**:

### **v1-v16 (formulated/)**
- Iterative development of the core computer vision and mathematics
- Started with basic ArUco detection using simple "Top/Bottom" pairing
- Progressed to sophisticated 3D geometry with position-based marker identification
- Added GUI (Tkinter), logging, tilt detection, and camera calibration support

### **v17 (formulated/v17/)**
- Final Tkinter GUI version with advanced features
- Comprehensive configuration, real-time camera controls, movement detection
- Still uses legacy "Top/Bottom" terminology in some places

### **Modbus Communication (Final Production Version)**
- **Current recommended version**
- Updated to use new position-based naming: **LT1, LT2, LB1, LB2, RT1, RT2, RB1, RB2**
- "L1-L2 pair" (Left side) and "R1-R2 pair" (Right side) terminology
- Flask-based web dashboard (no GUI dependencies)
- Industrial Modbus TCP integration with PLCs
- Headless operation optimized for Raspberry Pi deployment
- Real-time web interface accessible from control room PCs

---

## 🛠️ Prerequisites

**Virtual Environment (Required for ALL versions):**

1. **Create virtual environment** (run once):
   ```bash
   python -m venv venv
   ```

2. **Always activate before running any commands:**
   - **Windows**: `venv\Scripts\activate`
   - **Raspberry Pi / Linux**: `source venv/bin/activate`

3. **Install system dependencies** (Raspberry Pi):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-picamera2 python3-tk libatlas-base-dev
   ```

**Hardware Requirements:**
- Raspberry Pi with Camera Module
- Compatible ArUco markers (physical size must match configuration)
- Chessboard pattern for camera calibration (10x7 corners, 15mm squares)

---


## 🚀 Final Version (Recommended)

**⚠️ Important**: Make sure you have followed the [Prerequisites](#prerequisites) section above (virtual environment setup and system dependencies).

### Quick Start - Modbus Communication Version

```bash
# 1. Make sure your virtual environment is activated:
#    Windows: venv\Scripts\activate
#    Raspberry Pi/Linux: source venv/bin/activate

# 2. Configure the system (first time only)
cp modbus_communication/config.json .
# Edit config.json with your PLC IP, marker sizes, thresholds, etc.

# 3. Run the final Modbus version
cd modbus_communication
python app.py
```

The application will start a web server (default: http://localhost:5000) and two background threads:
- **Gap Engine**: Computer vision measurement using ArUco markers
- **Modbus Worker**: Communicates measurements to PLC via Modbus TCP

---

## 📁 Project Structure

```
rpi_gap_measurement_through_aruco_march/
├── README.md                          # This file - Main documentation
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── modbus_communication/              # FINAL PRODUCTION VERSION (Recommended)
│   ├── app.py                         # Main Flask web application
│   ├── gap_engine.py                  # Core computer vision & measurement logic
│   ├── modbus_worker.py               # PLC communication via Modbus TCP
│   ├── simulation_app.py              # Simulation/testing dashboard
│   ├── config.json                    # Configuration (PLC IP, marker sizes, etc.)
│   └── idea.excalidraw                # System architecture diagram
│
├── formulated/                        # Development versions (v1-v17)
│   ├── v17/                           # Latest Tkinter GUI version
│   ├── v1.py - v16.py                 # Historical iterations
│   ├── constants.py                   # Shared constants and styling
│   ├── measurement_logic.py           # Core measurement algorithms
│   ├── log.py                         # CSV + SQLite logging
│   ├── main.py                        # Entry point for v17
│   ├── app.py, logic.py, etc.         # v17 modular components
│   ├── gap_measurements.csv           # Historical measurement data
│   └── *.excalidraw                   # Block diagrams
│
├── calibration/                       # Camera calibration tools
│   ├── calibrate.py                   # Chessboard calibration script
│   ├── capture_images.py              # Image capture utility
│   └── camera_params*.npz             # Calibration matrices (generated)
│
├── dist_btw_2_pairs_of_aruco/         # Early dual-pair experiments
│   └── v1.py, v2.py, v3_two_corners.py
│
└── calibration/camera_params*.npz     # Camera calibration files
```

---

## 📊 Version History & Changes

### Major Milestones:

**Early Development (v1-v7):**
- Basic ArUco marker detection
- Single pair distance measurement
- Initial GUI development
- Basic logging implementation

**Mid Development (v8-v13):**
- Dual-pair support with position-based identification (precursor to LT/LB/RT/RB naming)
- Advanced 3D geometric intersection mathematics
- Tilt/rotation angle calculation (Euler angles)
- Improved coordinate system handling
- Multiple camera calibration support ("Left" and "Right" fixed side)

**Advanced Features (v14-v16):**
- Intelligent math fallback (Laser Hit vs Edge-Midpoint methods)
- Real-time tilt safety alerts (visual + color coding)
- Comprehensive telemetry dashboard
- Movement detection algorithms
- Performance optimizations and buffering

**v17 (GUI Version):**
- Complete modular architecture (app, logic, gui_widgets, utils, camera)
- Advanced camera controls (exposure, gain, white balance, etc.)
- Real-time movement state detection
- Enhanced configuration system
- Professional control-room aesthetic

**Final Modbus Version (Current):**
- **Headless web dashboard** (Flask + modern dark UI)
- **Industrial integration** via Modbus TCP to PLC
- Real-time web interface for control systems
- Shared state architecture between threads
- Production-ready error handling and reconnection logic
- **Removes Tkinter dependency** for better Raspberry Pi deployment

**Key Technical Improvements in Final Version:**
- Separation of concerns (gap_engine, modbus_worker, web app)
- Robust thread-safe shared data model
- Automatic PLC reconnection with heartbeat
- Configurable thresholds for rotation, pitch, yaw
- Support for both "Left" and "Right" fixed camera calibration
- Modern web-based UI accessible from any device on the network

---

## 🛠️ Technical Details

### Measurement Method
The system uses **ray-plane intersection** mathematics with position-based marker identification:

1. Detects up to 8 ArUco markers positioned as: **LT1, LT2, LB1, LB2** (Left side) and **RT1, RT2, RB1, RB2** (Right side)
2. Groups into **L1-L2 pair** (Left side markers) and **R1-R2 pair** (Right side markers)
3. Calculates 3D positions using camera calibration
4. Determines plane orientations from marker corners
5. Computes intersection point between opposing marker planes
6. Applies intelligent fallback for high-rotation scenarios

### Modbus Register Mapping
- **Registers 0-1**: Left side distance (L1-L2 pair, Float32)
- **Registers 2-3**: Right side distance (R1-R2 pair, Float32)
- **Register 4**: Error code
- **Register 5**: Heartbeat counter

### Configuration Parameters
- `marker_size_LT`, `marker_size_LB`, `marker_size_RT`, `marker_size_RB`: Physical marker sizes per position (critical for accuracy)
- `plc_ip/port`: Target PLC address
- `rot_threshold`, `pitch_threshold`, `yaw_threshold`: Alert thresholds
- `fixed_side`: "Left" or "Right" (selects calibration file)
- `rpi_id`: Unique identifier for the measurement node

---

## 📸 Calibration

**Critical for accuracy** - see [`calibration/README.md`](calibration/README.md) for detailed instructions.

1. Print a chessboard pattern
2. Capture 15+ images from different angles using `capture_images.py`
3. Run `calibrate.py` to generate `camera_params.npz`
4. Place calibration files in the root or calibration/ directory

**Note**: The system falls back to a generic pinhole model if calibration files are missing, but accuracy will be reduced.

---

## 📋 Requirements

**All commands must be run with the virtual environment activated** (see [Prerequisites](#prerequisites) section).

See [`requirements.txt`](requirements.txt):
- `numpy`, `opencv-contrib-python`, `Pillow`, `picamera2`
- `flask` (for Modbus version)
- `pyModbusTCP` (for PLC communication)

**Raspberry Pi specific system packages:**
```bash
sudo apt-get install python3-picamera2 python3-tk libatlas-base-dev
```

---

## 🔧 Troubleshooting

**Common Issues:**
- **"Command not found" or missing modules**: Ensure virtual environment is activated (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on Raspberry Pi)
- **Camera not detected**: Ensure `picamera2` is properly installed and enabled
- **Poor accuracy**: Verify marker size configuration and camera calibration
- **PLC connection failures**: Check IP address, port, and network connectivity
- **High CPU usage**: Adjust frame rate or resolution in the gap engine

**Logs:**
- Web console output
- `gap_measurements.csv` and `gap_measurements.db` in formulated/
- SQLite database for historical measurements

---

## 📄 Additional Documentation

- [`modbus_communication/README.md`](modbus_communication/README.md) - Final version details
- [`formulated/README.md`](formulated/README.md) - Development history
- [`calibration/README.md`](calibration/README.md) - Camera calibration guide
- `modbus_communication/idea.excalidraw` - System architecture diagram

---

**Built for industrial precision measurement applications.**

**Marker Naming Convention**: LT1/LT2/LB1/LB2 (Left side) and RT1/RT2/RB1/RB2 (Right side) enabling clear "L1-L2 pair" and "R1-R2 pair" references.

*Last updated: April 2026*
