import threading
import json
import os
import time
from flask import Flask, render_template_string, request, jsonify

# Import our custom modules
from gap_engine import run_gap_engine
from modbus_worker import run_modbus_client

app = Flask(__name__)

# --- 1. SHARED STATE ---
CONFIG_FILE = "config.json"

# Load initial config from file
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config_data = json.load(f)
else:
    # Defaults
    # NOTE: New naming convention uses position-based markers:
    # LT = Left Top, LB = Left Bottom, RT = Right Top, RB = Right Bottom
    # This enables referring to "L1-L2 pair" (Left side markers) and "R1-R2 pair" (Right side markers)
    config_data = {
        "rpi_id": "UNIT_RPI_01",
        "plc_ip": "192.168.1.50",
        "plc_port": 502,
        "marker_size_top": 100.0,      # Legacy name (maps to LT)
        "marker_size_bot": 100.0,      # Legacy name (maps to LB)
        "marker_size_LT": 100.0,       # Left Top markers
        "marker_size_LB": 100.0,       # Left Bottom markers
        "marker_size_RT": 100.0,       # Right Top markers
        "marker_size_RB": 100.0,       # Right Bottom markers
        "logging_enabled": True,
        "focus_value": 0,
        "rot_threshold": 5.0,
        "pitch_threshold": 6.0,
        "yaw_threshold": 4.0,
        "fixed_side": "Left"
    }

# This dictionary is shared between ALL THREADS
# NOTE: While internal keys still use "top_dist"/"bottom_dist" for compatibility with existing code,
# the semantic meaning has been updated to represent:
# - "top_dist"  → Left side measurement (L1-L2 pair: LT + LB markers)
# - "bottom_dist" → Right side measurement (R1-R2 pair: RT + RB markers)
shared_data = {
    "top_dist": 0.0,      # Left side (L1-L2 pair) distance
    "bottom_dist": 0.0,   # Right side (R1-R2 pair) distance
    "error_code": 0,
    "plc_online": False,
    "brightness": 0,
    "status": "Initializing..."
}

# --- 2. WEB ENDPOINTS ---

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>RPi ArUco | Industrial Dash</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0b0e14; color: #e0e6ed; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        .nav { background: #151b23; padding: 15px 30px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
        .main { display: flex; flex: 1; padding: 20px; gap: 20px; }
        .card { background: #151b23; border-radius: 12px; border: 1px solid #30363d; padding: 25px; flex: 1; transition: transform 0.2s; }
        .value { font-size: 56px; font-weight: bold; color: #00ff88; margin: 10px 0; }
        .unit { font-size: 18px; color: #8b949e; }
        .status-box { padding: 10px 20px; border-radius: 50px; font-size: 14px; font-weight: bold; }
        .online { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; }
        .offline { background: rgba(255, 69, 0, 0.1); color: #ff4500; border: 1px solid #ff4500; }
        .settings-panel { width: 350px; background: #151b23; border-left: 1px solid #30363d; padding: 25px; overflow-y: auto; }
        label { display: block; margin: 15px 0 5px; color: #8b949e; font-size: 12px; }
        input { width: 100%; padding: 10px; background: #0b0e14; border: 1px solid #30363d; color: white; border-radius: 6px; }
        button { width: 100%; padding: 12px; background: #00ff88; color: #0b0e14; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; margin-top: 20px; }
        button:hover { background: #00cc6e; }
    </style>
</head>
<body>
    <div class="nav">
        <div style="font-size: 20px; font-weight: bold;">NODE: <span id="rpi_id_display">...</span></div>
        <div id="plc_status" class="status-box offline">PLC DISCONNECTED</div>
    </div>
    
    <div class="main">
        <div class="card">
            <h1>LEFT SIDE (L1-L2 PAIR)</h1>
            <div class="value" id="top_dist">0.000 <span class="unit">mm</span></div>
            <p style="color: #fb923c; font-size: 14px;">LT1, LT2, LB1, LB2 markers</p>
            <p id="error_text" style="color: #ff4500;"></p>
        </div>
        <div class="card">
            <h1>RIGHT SIDE (R1-R2 PAIR)</h1>
            <div class="value" id="bot_dist">0.000 <span class="unit">mm</span></div>
            <p style="color: #c084fc; font-size: 14px;">RT1, RT2, RB1, RB2 markers</p>
        </div>

        <div class="settings-panel">
            <h2>Configuration</h2>
            <label>PLC IP ADDRESS</label>
            <input type="text" id="plc_ip" value="">

            <label>MARKER SIZE LT (Left Top / mm)</label>
            <input type="number" id="size_top" value="">

            <label>MARKER SIZE LB (Left Bottom / mm)</label>
            <input type="number" id="size_bot" value="">

            <label>MARKER SIZE RT (Right Top / mm)</label>
            <input type="number" id="size_rt" value="">

            <label>MARKER SIZE RB (Right Bottom / mm)</label>
            <input type="number" id="size_rb" value="">
            
            <label>LENS FOCUS STEP</label>
            <input type="range" id="focus" min="0" max="1000" style="width: 100%;">
            
            <button onclick="saveSettings()">Apply Changes</button>
            <p style="font-size: 11px; color: #666; margin-top: 15px;">Note: Changes update config.json and live threads instantly.</p>
        </div>
    </div>

    <script>
        function updateUI() {
            fetch('/get_live_data')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('top_dist').childNodes[0].nodeValue = data.top_dist.toFixed(3) + " ";
                    document.getElementById('bot_dist').childNodes[0].nodeValue = data.bottom_dist.toFixed(3) + " ";
                    
                    const plcBox = document.getElementById('plc_status');
                    if (data.plc_online) {
                        plcBox.innerText = "PLC ONLINE";
                        plcBox.className = "status-box online";
                    } else {
                        plcBox.innerText = "PLC OFFLINE";
                        plcBox.className = "status-box offline";
                    }
                });
        }

        function loadConfig() {
            fetch('/get_config')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('rpi_id_display').innerText = data.rpi_id;
                    document.getElementById('plc_ip').value = data.plc_ip;
                    document.getElementById('size_top').value = data.marker_size_LT || data.marker_size_top;
                    document.getElementById('size_bot').value = data.marker_size_LB || data.marker_size_bot;
                    // Note: size_rt and size_rb inputs would need backend support
                    document.getElementById('focus').value = data.focus_value;
                });
        }

        function saveSettings() {
            const payload = {
                plc_ip: document.getElementById('plc_ip').value,
                marker_size_top: parseFloat(document.getElementById('size_top').value),      // LT
                marker_size_bot: parseFloat(document.getElementById('size_bot').value),    // LB
                // Note: RT and RB marker sizes would be added in a full implementation
                focus_value: parseInt(document.getElementById('focus').value)
            };
            fetch('/update_config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            }).then(() => alert("Settings Persistent & Applied!\n\nNOTE: Using new LT/LB/RT/RB naming convention."));
        }

        setInterval(updateUI, 200);
        loadConfig();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/get_live_data')
def get_live_data():
    return jsonify(shared_data)

@app.route('/get_config')
def get_config():
    return jsonify(config_data)

@app.route('/update_config', methods=['POST'])
def update_config():
    global config_data
    new_data = request.json
    # Update RAM
    config_data.update(new_data)
    # Update DISK (Persistent)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=2)
    return jsonify({"status": "success"})

# --- 3. MAIN SERVICE START ---

if __name__ == '__main__':
    # Thread A: The ArUco Math Engine
    engine_thread = threading.Thread(
        target=run_gap_engine, 
        args=(shared_data, config_data), 
        daemon=True
    )
    
    # Thread B: The Modbus PLC Client (Push)
    modbus_thread = threading.Thread(
        target=run_modbus_client, 
        args=(shared_data, config_data), 
        daemon=True
    )
    
    print("[MAIN] Starting Subroutines...")
    engine_thread.start()
    modbus_thread.start()
    
    print("[MAIN] Starting Dashboard Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)
