import tkinter as tk
from app import MeasurementApp
import sys
import os

# Ensure the v17 directory is in path for imports to work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
