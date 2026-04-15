# Formulated Development Versions

This directory contains the **iterative development history** of the ArUco Gap Measurement System, from initial experiments to the mature v17 GUI version.

## 📈 Evolution Summary

The project progressed through **17 major versions**, each building upon the lessons of the previous:

### Phase 1: Foundation (v1-v5)
- **v1-v3**: Basic ArUco marker detection and single-pair distance calculation
- **v4**: Added comprehensive CSV logging (`v4_log_to_csv.py`)
- **v5**: Finalized basic logging architecture (`v5_final_log.py`)
- **Focus**: Getting reliable marker detection and basic measurements working

### Phase 2: Dual-Pair Architecture (v6-v10)
- **v6**: First GUI implementation (`v6_gui.py.py`)
- **v7**: Dual-pair support (`v7_two_pairs.py`)
- **v8**: Improved coordinate handling and mathematics
- **v9**: Angle correction and tilt detection (`v9_angle_correction.py`)
- **v10**: Multi-side detection improvements (`v10_multiside.py`)
- **Focus**: Supporting two independent marker pairs with position-based identification (foundation for LT/LB/RT/RB naming)

### Phase 3: Advanced Mathematics & Telemetry (v11-v15)
- **v11-v13**: Iterative improvements to 3D geometry calculations
- **v14**: Major refactoring of measurement logic
- **v15**: Enhanced telemetry and data visualization
- **Focus**: Sophisticated 3D intersection mathematics, Euler angle calculations, and comprehensive telemetry

### Phase 4: Production Polish (v16-v17)
- **v16**: Performance optimizations, buffering, and stability improvements
- **v17**: **Complete modular rewrite** with professional architecture:
  - `main.py` - Entry point
  - `app.py` - Tkinter GUI framework with tabs (Live Monitor, Telemetry, Configuration, Camera)
  - `logic.py` - Core measurement engine (ported from v16)
  - `camera.py` - Camera management and calibration
  - `gui_widgets.py` - Reusable smart UI components
  - `utils.py` - Helper functions and calibration loading
  - `config.py` - Constants, colors, and styling
  - `log.py` - Enhanced logging with rotation data

## 📋 Key Technical Improvements

### Mathematical Evolution
1. **Basic distance** → **3D coordinate transformation**
2. **Single pair** → **Dual independent pairs** with position-based naming (LT1/LT2, LB1/LB2, RT1/RT2, RB1/RB2)
3. **"Top/Bottom" pairing** → **"L1-L2" and "R1-R2" side-based pairing**
4. **Simple averaging** → **Intelligent buffering + outlier rejection**
5. **Basic intersection** → **Ray-plane intersection with fallback logic**
6. **No orientation** → **Full Euler angle calculation** (Roll, Pitch, Yaw)

### Safety & Usability
- Visual tilt warnings (red boxes, yellow text)
- Automatic math mode switching (>20° rotation triggers fallback)
- Real-time camera parameter adjustment
- Movement state detection (initial vs final positions)

### Data Architecture
- Comprehensive CSV logging (`gap_measurements.csv`)
- SQLite database backup
- Timestamped image capture (`captured_images/`)
- 30+ data points per measurement (coordinates, angles, metadata)

## 🧪 Important Files

- **`constants.py`** - Shared configuration, colors, and parameters
- **`measurement_logic.py`** - Port of the v16/v17 core measurement engine
- **`log.py`** - Robust logging with 30+ columns including rotation data
- **`v17/main.py`** - Clean entry point for the latest GUI version
- **`v17/app.py`** - Main application class with full feature set
- **`block_diagram*.excalidraw`** - System architecture visualizations
- **`gap_measurements.csv`** - Historical measurement data

## 🚀 Running Development Versions

### Virtual Environment Setup (Required for all versions)

```bash
# From project root:

# Create virtual environment (if not already created)
python -m venv venv

# Activate the virtual environment:
# Windows:
venv\Scripts\activate
# Raspberry Pi / Linux:
# source venv/bin/activate

# Install dependencies (if not already installed)
pip install --upgrade pip
pip install -r ../requirements.txt
```

### Running the Applications

```bash
# Make sure your virtual environment is activated first:
# Windows: venv\Scripts\activate
# Raspberry Pi/Linux: source venv/bin/activate

cd formulated

# Latest GUI version (v17) - Recommended for development
python v17/main.py

# Or run specific historical versions
python v16.py
python v15.py
# etc.
```

**Note**: Most early versions (`v1.py` through `v13.py`) were experimental and may require modifications to run with current dependencies. The virtual environment ensures all versions use the same consistent dependency set.

## 📊 What Was Learned

1. **Marker size is critical** - Must match physical markers exactly (now per position: LT, LB, RT, RB)
2. **Camera calibration dramatically improves accuracy**
3. **Position-based identification** (LT1/LT2, LB1/LB2, RT1/RT2, RB1/RB2) is clearer than simple Top/Bottom
4. **Side-based pairing** ("L1-L2 pair" and "R1-R2 pair") provides better semantic meaning than "Top pair"
5. **Tilt detection is essential** for operational safety
6. **Modular architecture** (v17) is much more maintainable than monolithic scripts
7. **Web interface** (final version) is more suitable for industrial deployment than Tkinter GUI

---

## ⬇️ Final Version

The **production version** has been moved to `../modbus_communication/` and has been updated to use the new position-based naming convention:

- **LT1, LT2, LB1, LB2, RT1, RT2, RB1, RB2** marker identification
- **"L1-L2 pair"** (Left side) and **"R1-R2 pair"** (Right side) terminology
- Flask web dashboard (no Tkinter dependency)
- Modbus TCP integration with industrial PLCs
- Headless operation optimized for Raspberry Pi
- Modern web-based control room interface

See the root [`README.md`](../README.md) and [`modbus_communication/README.md`](../modbus_communication/README.md) for complete details on the new naming convention.

---

*This directory preserves the complete development history for reference, learning, and potential future improvements.*
