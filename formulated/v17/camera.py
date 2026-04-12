import cv2 as cv
import numpy as np

class CameraManager:
    """
    Manages the Raspberry Pi camera initialization and controls.
    """
    def __init__(self, resolution):
        """
        Initialize the camera manager.
        
        Args:
            resolution (tuple): (width, height) tuple.
        """
        self.resolution = resolution
        self.pc = None
        self.AWB_MODES = {"Auto": 0, "Tungsten": 1, "Fluorescent": 2, "Indoor": 3, "Daylight": 4, "Cloudy": 5}

    def start(self):
        """
        Start the Picamera2 instance.
        
        Returns:
            Picamera2: The started camera instance or None if failed.
        """
        try:
            from picamera2 import Picamera2
            self.pc = Picamera2()
            self.pc.configure(self.pc.create_video_configuration(main={"size": self.resolution, "format": "RGB888"}))
            self.pc.start()
            return self.pc
        except Exception as e:
            print(f"[CAMERA INIT ERROR] {e}")
            return None

    def capture_frame(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            numpy.ndarray: BGR frame or None if failed.
        """
        if not self.pc: return None
        try:
            return cv.cvtColor(self.pc.capture_array(), cv.COLOR_RGB2BGR)
        except Exception:
            return None

    def apply_controls(self, app_vars):
        """
        Apply control variables (exposure, gain, etc.) to the camera.
        
        Args:
            app_vars: Struct/Dict containing BooleanVar and DoubleVar camera settings.
        """
        if not self.pc: return
        
        controls = {}
        ae_on = app_vars.cam_ae.get()
        controls["AeEnable"] = ae_on
        if not ae_on:
            controls["ExposureTime"] = int(app_vars.cam_exposure.get())
            controls["AnalogueGain"] = float(app_vars.cam_gain.get())
        
        awb_on = app_vars.cam_awb.get()
        controls["AwbEnable"] = awb_on
        if not awb_on:
            controls["AwbMode"] = self.AWB_MODES.get(app_vars.cam_awb_mode.get(), 0)
        
        controls["Brightness"] = float(app_vars.cam_brightness.get())
        controls["Contrast"] = float(app_vars.cam_contrast.get())
        controls["Saturation"] = float(app_vars.cam_saturation.get())
        controls["Sharpness"] = float(app_vars.cam_sharpness.get())
        
        try:
            self.pc.set_controls(controls)
        except Exception as e:
            print(f"[CAMERA CONTROL ERROR] {e}")
