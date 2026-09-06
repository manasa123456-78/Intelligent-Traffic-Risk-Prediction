import cv2
import numpy as np
from config import CFG

VIDEO_PATH = CFG.VIDEO_PATH

points = []
window_name = 'Click 4 points for ROI (Press Q to exit)'

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point {len(points)}: [{x}, {y}]")

        # Draw a red dot where you clicked
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

        # Connect the dots if 4 points are selected
        if len(points) == 4:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 0), 3)
           
            print("ROI_POLYGON = np.array([")
            for p in points:
                print(f"    [{p[0]}, {p[1]}],")
            print("])")
            print("="*50)

        cv2.imshow(window_name, img)

# Load the first frame of the video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, img = cap.read()
cap.release()

if not ret:
    print(f"Error: Could not read {VIDEO_PATH}. Check the filename.")
    exit()

print("INSTRUCTIONS:")
print("1. Click 4 points on the image to outline your tracking zone.")
print("2. Press the 'q' key on your keyboard to close the window when finished.")

# --- THE FIX: Make the window fit on your screen ---
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allows the window to be resized
cv2.resizeWindow(window_name, 1280, 720)         # Forces it to a standard screen size

cv2.imshow(window_name, img)
cv2.setMouseCallback(window_name, click_event)

# Wait for the user to press 'q'
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
