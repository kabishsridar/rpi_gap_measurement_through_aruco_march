import tkinter as tk
import sys
import os

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import MeasurementApp

def main():
    """
    Main entry point for the ArUco Measurement System v17.
    """
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
