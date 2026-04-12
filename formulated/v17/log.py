import csv
import sqlite3
import os
from datetime import datetime

DB_NAME = "gap_measurements.db"
CSV_NAME = "gap_measurements.csv"

def init_log():
    """
    Initialize CSV and SQLite database for logging.
    """
    if not os.path.exists(CSV_NAME):
        with open(CSV_NAME, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Distance_mm", "Ratio_k", "A_z", "X_z", "TR_z", "BR_z", "B_z", "C_z"])
    
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS measurements 
                 (timestamp TEXT, distance REAL, ratio_k REAL, A_z REAL, X_z REAL)''')
    conn.commit()
    conn.close()

def record(dist, k, a, x, tr, br, b, c, *args):
    """
    Record a measurement entry to both CSV and SQLite.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_NAME, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ts, f"{dist:.3f}", f"{k:.4f}", f"{a[2]:.2f}", f"{x[2]:.2f}", f"{tr[2]:.2f}", f"{br[2]:.2f}", f"{b[2]:.2f}", f"{c[2]:.2f}"])
    
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO measurements VALUES (?, ?, ?, ?, ?)", (ts, dist, k, a[2], x[2]))
    conn.commit()
    conn.close()
