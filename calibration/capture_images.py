import cv2, os, time
from datetime import datetime
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except:
    PICAMERA2_AVAILABLE = False

# --- SETTINGS FOR YOUR BOARD (CONFIRMED) ---
CORNERS = (10, 7) 
SAVE_DIR = "board_images"
# ------------------------------------------

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

def main():
    picam = Picamera2()
    # High quality 720p stream
    config = picam.create_video_configuration(main={"format": "RGB888", "size": (1280, 720)})
    picam.configure(config)
    picam.start()

    print(f"Goal: {CORNERS} internal corners")
    print("Controls: SPACE to save, 'q' to quit")

    count = 0
    try:
        while True:
            frame_rgb = picam.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Robust detection settings
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, CORNERS, flags)
            
            display = frame.copy()
            if ret:
                cv2.drawChessboardCorners(display, CORNERS, corners, ret)
                cv2.putText(display, f"Size: {CORNERS} | SAVED: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Board Capture", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '):
                count += 1
                path = os.path.join(SAVE_DIR, f"board_{count:03d}.jpg")
                cv2.imwrite(path, frame)
                print(f"Saved {path}")
    finally:
        picam.stop()

if __name__ == "__main__":
    main()
