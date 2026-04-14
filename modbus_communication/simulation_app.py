import threading
import time
from flask import Flask, render_template_string

app = Flask(__name__)

# Shared Data Structure (Simulating your config and status)
shared_data = {"counter": 0, "status": "Initializing"}

def background_worker():
    """
    Simulation of the ArUco Engine.
    This runs in a separate thread and never stops.
    """
    global shared_data
    while True:
        shared_data["counter"] += 1
        shared_data["status"] = "Active - Measuring Gap..."
        # Simulate local console print
        # print(f" >>> [Background Thread] Processing... Count: {shared_data['counter']}")
        time.sleep(1) # Increase every 1 second

# Modern Dashboard with Auto-Update logic
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>RPi ArUco Dashboard</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #0f0f0f, #1a1a1a); 
            color: white; 
            height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .container { 
            background: rgba(255, 255, 255, 0.05); 
            padding: 40px; 
            border-radius: 24px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        h1 { margin: 0; font-weight: 300; letter-spacing: 2px; color: #888; font-size: 14px; text-transform: uppercase; }
        #counter { font-size: 120px; font-weight: bold; margin: 20px 0; color: #00ffa2; text-shadow: 0 0 20px rgba(0,255,162,0.3); }
        .status-badge { 
            background: rgba(0, 255, 162, 0.1); 
            color: #00ffa2; 
            padding: 8px 16px; 
            border-radius: 50px; 
            font-size: 12px; 
            border: 1px solid rgba(0,255,162,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Live ArUco Measurement</h1>
        <div id="counter">0</div>
        <div class="status-badge" id="status">Waiting for Thread...</div>
        <p style="color: #666; font-size: 12px; margin-top: 30px;">SIMULATED REAL-TIME DATA (NO REFRESH NEEDED)</p>
    </div>

    <script>
        // THIS IS THE BRAINS OF THE AUTO-UPDATE
        function fetchData() {
            fetch('/get_data') // Ask Flask for data
                .then(response => response.json())
                .then(data => {
                    // Update the text on the screen instantly
                    document.getElementById('counter').innerText = data.counter;
                    document.getElementById('status').innerText = data.status;
                });
        }

        // Run this every 200 milliseconds (5 times per second)
        setInterval(fetchData, 200);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(dashboard_html)

@app.route('/get_data')
def get_data():
    """API Endpoint for JavaScript to fetch the shared thread data"""
    return {
        "counter": shared_data["counter"],
        "status": shared_data["status"]
    }

if __name__ == '__main__':
    # Start Background Thread
    threading.Thread(target=background_worker, daemon=True).start()

    # Start Flask Server
    app.run(host='0.0.0.0', port=5000, debug=False)
