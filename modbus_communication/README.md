# Modbus Communication Version - FINAL PRODUCTION

**Industrial RPi ArUco Gap Measurement System with PLC Integration**

This is the **final, production-ready version** of the gap measurement system. It replaces the Tkinter GUI with a modern web dashboard and adds robust Modbus TCP communication for integration with industrial PLCs.

---

## 🎯 Overview

This version is designed for **industrial deployment** on Raspberry Pi:

- **Web-based dashboard** accessible from any device on the network
- **Real-time Modbus TCP** communication with PLC systems
- **Headless operation** (no GUI dependencies, better for embedded use)
- **Production-grade error handling** and reconnection logic
- **Thread-safe shared state** architecture

---

## 🚀 Quick Start

**⚠️ Important**: Make sure you have followed the [Prerequisites section in the main README](../README.md#prerequisites) (virtual environment setup and system dependencies).

### Run the Application

```bash
# 1. Make sure your virtual environment is activated:
#    Windows: venv\Scripts\activate
#    Raspberry Pi/Linux: source venv/bin/activate

# 2. Copy and configure (first time only)
cp modbus_communication/config.json .

# 3. Edit config.json with your settings:
#    - plc_ip (your PLC address)
#    - marker_size_top/bot (physical marker size in mm)
#    - thresholds for tilt detection
#    - rpi_id for identification

# 4. Run the final Modbus version
cd modbus_communication
python app.py
```

The system will start:
1. **Web server** on port 5000 (http://your-rpi-ip:5000)
2. **Gap Engine thread** - Computer vision measurement
3. **Modbus Client thread** - PLC communication

---

## 📋 Architecture

### Three Main Components

#### 1. `app.py` - Web Dashboard
- Flask application with modern dark industrial UI
- Real-time updates via JavaScript fetch
- Live display of:
  - Upper and Lower sensor readings (mm)
  - PLC connection status
  - Error codes and messages
  - Brightness level
  - Configuration controls

#### 2. `gap_engine.py` - Computer Vision Core
- **Strict port of v16/v17 measurement logic**
- Uses `picamera2` for high-performance capture
- ArUco marker detection with `opencv-contrib-python`
- Dual-pair measurement (Top + Bottom isolation)
- Intelligent tilt detection and math fallback
- Updates the shared data dictionary in real-time

#### 3. `modbus_worker.py` - PLC Integration
- Connects to PLC using `pyModbusTCP`
- Converts measurements to Modbus registers
- Implements heartbeat and auto-reconnect logic
- Robust error handling for industrial environments

---

## ⚙️ Configuration (`config.json`)

```json
{
  "rpi_id": "UNIT_RPI_01",
  "plc_ip": "192.168.1.50",
  "plc_port": 502,
  "marker_size_top": 53.8,
  "marker_size_bot": 53.8,
  "logging_enabled": false,
  "focus_value": 0,
  "rot_threshold": 12.0,
  "pitch_threshold": 12.0,
  "yaw_threshold": 12.0,
  "fixed_side": "Left"
}
```

**Key Parameters:**
- `marker_size_*`: **Critical** - must match physical marker size in millimeters
- `fixed_side`: Choose "Left" or "Right" to select appropriate calibration file
- `rot/pitch/yaw_threshold`: Degrees that trigger warning states
- `plc_ip/port`: Target PLC for data transmission

---

## 📡 Modbus Register Map

| Register | Content              | Type    | Description |
|----------|----------------------|---------|-------------|
| 0-1      | Top Distance         | Float32 | Upper gap measurement (mm) |
| 2-3      | Bottom Distance      | Float32 | Lower gap measurement (mm) |
| 4        | Error Code           | Int16   | 0 = OK, >0 = error condition |
| 5        | Heartbeat            | Int16   | Incrementing counter |

**Error Codes:**
- `0`: Normal operation
- `1-10`: Marker detection issues
- `11-20`: Tilt/rotation warnings
- `99`: Camera/hardware errors

---

## 🔬 Core Measurement Technology

The `gap_engine.py` implements the mature v16/v17 algorithms:

1. **ArUco Detection**: 4x4_50 dictionary, 1280x720 resolution
2. **Camera Calibration**: Loads `camera_params.npz` from `../calibration/`
3. **Dual-Pair Logic**: Top markers only pair with top, bottom with bottom
4. **3D Reconstruction**: Converts 2D corners to 3D world coordinates
5. **Geometric Intersection**: Calculates gap using ray-plane mathematics
6. **Tilt Analysis**: Computes Euler angles for safety monitoring
7. **Fallback Logic**: Switches mathematical approach for high-rotation cases

**Shared Data Structure:**
```python
shared_data = {
    "top_dist": 0.0,
    "bottom_dist": 0.0,  # Note: also accessible as "bot_dist"
    "error_code": 0,
    "plc_online": False,
    "brightness": 0,
    "status": "Running..."
}
```

---

## 🌐 Web Dashboard Features

- **Real-time updating** gauges for both sensors
- **PLC connectivity status** with color coding
- **Live error messaging** and diagnostics
- **Configuration panel** (editable at runtime)
- **Responsive industrial design**
- **Mobile-friendly** interface

Access at: `http://<raspberry-pi-ip>:5000`

---

## 🛠️ Additional Tools

### `simulation_app.py`
A standalone simulation dashboard for testing the web interface without camera or PLC dependencies. Useful for UI development and demonstration.

---

## 📁 File Dependencies

- **Calibration files**: `../calibration/camera_params*.npz` (highly recommended)
- **Configuration**: `config.json` (auto-created with defaults if missing)
- **Logging**: Uses the logging system from `../formulated/log.py`

---

## 🔧 Troubleshooting

**Camera Issues:**
- Ensure `picamera2` is enabled: `sudo raspi-config`
- Check camera connection and permissions
- Verify `camera_params.npz` exists

**Modbus/PLC Issues:**
- Verify network connectivity to PLC
- Check that Modbus TCP port 502 is open
- Confirm PLC is configured to accept connections from the RPi

**Performance Issues:**
- Reduce resolution in `gap_engine.py` if needed
- Check CPU temperature and throttling
- Adjust collection buffer size (`COLLECT_N`)

**Accuracy Issues:**
- Verify marker sizes in config exactly match physical markers
- Ensure proper camera calibration
- Check lighting conditions (avoid extreme brightness variation)

---

## 📈 Compared to Previous Versions

**Advantages over v17 GUI:**
- No Tkinter dependency (lighter, more stable on RPi)
- Web interface accessible from control room PCs
- Native industrial protocol integration (Modbus)
- Better suited for 24/7 embedded operation
- Modern, responsive UI design

**Preserves all mathematical intelligence** from the v16/v17 development iterations.

---

See the main [`../README.md`](../README.md) for complete project documentation, version history, and calibration instructions.

**This is the recommended version for production deployment.**
