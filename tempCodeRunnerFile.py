import cv2
import numpy as np
import time
from collections import Counter

def run_detector(video_source=0):
    """
    Processes video to identify multiple traffic light states simultaneously.
    """
    # Define HSV color ranges
    color_ranges = {
        'red': ([0, 120, 70], [10, 255, 255]),
        'red_wrap': ([170, 120, 70], [180, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'green': ([40, 70, 70], [90, 255, 255]),
    }

    # Define BGR colors for drawing bounding boxes
    box_colors = {
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
    }

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    # For FPS calculation
    start_time = time.time()
    frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        # FPS Calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 💡 NEW: List to store all detections in the current frame
        all_detections = []
        combined_mask = np.zeros(frame.shape[:2], dtype="uint8")

        for color, (lower, upper) in color_ranges.items():
            if color == 'red_wrap': continue # Handled with 'red'

            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv_frame, lower, upper)
            
            if color == 'red':
                mask_wrap = cv2.inRange(hsv_frame, np.array(color_ranges['red_wrap'][0]), np.array(color_ranges['red_wrap'][1]))
                mask = cv2.add(mask, mask_wrap)
            
            kernel = np.ones((5, 5), np.uint8)
            mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
            
            combined_mask = cv2.add(combined_mask, mask_cleaned)

            contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 💡 NEW: Loop through ALL contours without breaking
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 500: continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0: continue
                
                circularity = 4 * np.pi * (area / (perimeter * perimeter))

                if 0.6 < circularity < 1.4:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Store the detection details instead of drawing immediately
                    all_detections.append({'color': color, 'box': (x, y, w, h)})
        
        # --- Drawing and Status Update Section ---
        
        # 💡 NEW: Update dashboard status based on all detections
        if not all_detections:
            status_text = "Searching..."
        else:
            # Count the number of lights of each color
            light_counts = Counter(d['color'] for d in all_detections)
            status_text = "Detected: " + ", ".join([f"{count} {color.capitalize()}" for color, count in light_counts.items()])
        
        # 💡 NEW: Draw all stored detections
        for detection in all_detections:
            color = detection['color']
            x, y, w, h = detection['box']
            draw_color = box_colors.get(color, (255, 0, 0))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            cv2.putText(frame, color.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
            
        # Draw the status dashboard
        dashboard_height = 60
        cv2.rectangle(frame, (0, frame.shape[0] - dashboard_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, f"Status: {status_text}", (10, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the diagnostic and main windows
        cv2.imshow('Diagnostic Mask', combined_mask)
        cv2.imshow('Traffic Light Detection - Final', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Main execution ---
if __name__ == '__main__':
    # Use 0 for webcam or provide a path to a video file
    video_source = 0 
    run_detector(video_source)