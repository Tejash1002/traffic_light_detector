import cv2
import numpy as np

# --- Configuration Section ---

# Define HSV color ranges for Red, Yellow, and Green.
# These are the most likely values you'll need to adjust for your specific video.
COLOR_RANGES = {
    'red': [
        (np.array([0, 120, 120]), np.array([10, 255, 255])),
        (np.array([170, 120, 120]), np.array([180, 255, 255]))
    ],
    'yellow': [
        (np.array([20, 100, 100]), np.array([35, 255, 255]))
    ],
    'green': [
        (np.array([40, 70, 70]), np.array([90, 255, 255]))
    ],
}

# Define BGR colors for drawing boxes, mapping directly to the color names
BOX_COLORS = {
    'red': (0, 0, 255),      # BGR for Red
    'yellow': (0, 255, 255), # BGR for Yellow
    'green': (0, 255, 0)     # BGR for Green
}

# --- Detection Parameters ---
# Adjust these values based on the video to improve accuracy
MIN_CONTOUR_AREA = 500  # Filters out detections that are too small (likely noise)
MIN_CIRCULARITY = 0.6   # A perfect circle has a circularity of 1
MAX_CIRCULARITY = 1.4
# Kernel for morphological operations (noise reduction).
# A larger kernel (e.g., (7,7)) removes more noise but can also remove small, distant lights.
MORPH_KERNEL = np.ones((7, 7), np.uint8)


# --- Main Detection Logic ---

def find_and_draw_lights(frame, hsv_frame):
    """
    Analyzes a single frame to find and draw traffic lights on the frame.

    Args:
        frame (np.array): The original BGR frame for drawing.
        hsv_frame (np.array): The HSV-converted frame for color detection.
    """
    detections = []

    for color_name, ranges in COLOR_RANGES.items():
        combined_mask = None
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv_frame, lower, upper)
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.add(combined_mask, mask)

        # Apply morphological opening to remove noise
        opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        
        contours, _ = cv2.findContours(opened_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            if MIN_CIRCULARITY < circularity < MAX_CIRCULARITY:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'box': (x, y, w, h),
                    'color': color_name,
                    'area': area
                })

    # If lights were detected, draw them all
    if detections:
        # Draw all valid detections found in the frame
        for detection in detections:
            x, y, w, h = detection['box']
            color_name = detection['color']
            box_color = BOX_COLORS.get(color_name, (255, 255, 255))

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            label = f"{color_name.capitalize()}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)


def detect_traffic_light(video_source=0):
    """
    Processes video from a camera or file to identify traffic light states.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    print("Starting video stream. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream. Looping video file.")
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Find lights and draw them on the frame
        find_and_draw_lights(frame, hsv_frame)

        # Display the final detection window
        cv2.imshow('Traffic Light Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video stream stopped.")

# --- Main Execution ---
if __name__ == '__main__':
    # --- IMPORTANT ---
    # To use your webcam, set video_source = 0
    # To use a video file, change the source to the file path.
    # For example: video_source = "C:/Users/YourUser/Videos/traffic.mp4"
    video_source = 0 
    detect_traffic_light(video_source)

