import tkinter as tk
from app import MeasurementApp

def main():
    root = tk.Tk()
    app = MeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
