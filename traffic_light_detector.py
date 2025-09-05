import cv2
import numpy as np

def detect_traffic_light(video_source=0):
    """
    Processes video from a camera or file to identify traffic light states.

    Args:
        video_source (int or str): The camera index (e.g., 0 for the default webcam) 
                                     or the path to a video file.
    """
    # 1. Define HSV color ranges for Red, Yellow, and Green
    color_ranges = {
        'red_1': (np.array([0, 120, 70]), np.array([10, 255, 255])),
        'red_2': (np.array([170, 120, 70]), np.array([180, 255, 255])),
        'yellow': (np.array([20, 100, 100]), np.array([35, 255, 255])), 
        'green': (np.array([40, 70, 70]), np.array([90, 255, 255])),
    }

    # Define BGR colors for drawing the boxes based on detected light state
    box_colors = {
        'red': (0, 0, 255),    # Red color in BGR
        'yellow': (0, 255, 255), # Yellow color in BGR
        'green': (0, 255, 0)   # Green color in BGR
    }
    
    # 2. Initialize video capture
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    while True:
        # 3. Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            if isinstance(video_source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # 4. Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_color = None
        current_box_color = (0, 255, 0) # Default to green if no specific color found yet

        # 5. Process each color to find a potential traffic light
        for color_key, (lower, upper) in color_ranges.items():
            # Create a mask for the current color
            if color_key == 'red_1':
                mask1 = cv2.inRange(hsv_frame, lower, upper)
                mask2 = cv2.inRange(hsv_frame, color_ranges['red_2'][0], color_ranges['red_2'][1])
                mask = cv2.add(mask1, mask2)
                color_name = 'red' # Use 'red' for labeling and box color lookup
            elif color_key == 'red_2':
                continue # Skip processing red_2 as it's handled with red_1
            else:
                mask = cv2.inRange(hsv_frame, lower, upper)
                color_name = color_key # Use the color_key directly (e.g., 'yellow', 'green')

            # 6. Find contours (shapes) in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # 7. Validate contours to find the best candidate for a traffic light
            for contour in contours:
                # Validation by size (area)
                area = cv2.contourArea(contour)
                if area < 500:  # Filter out small, noisy detections
                    continue

                # Validation by shape (approximating circularity)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * (area / (perimeter * perimeter))

                # We check for a reasonably circular shape
                if 0.7 < circularity < 1.3:
                    detected_color = color_name
                    current_box_color = box_colors.get(detected_color, (0, 255, 0)) # Get color, default to green
                    
                    # Draw a bounding box and label on the frame
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), current_box_color, 2)
                    label = f"Detected: {detected_color.capitalize()}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_box_color, 2)
                    break # Found a good candidate, stop checking other contours for this color
            
            if detected_color:
                break # Found a light, no need to check other colors

        # 8. Display the resulting frame
        cv2.imshow('Traffic Light Detection', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Release resources
    cap.release()
    cv2.destroyAllWindows()

# --- Main execution ---
if __name__ == '__main__':
    # To use your webcam, keep the source as 0
    # To use a video file, change the source to the file path, e.g., "traffic_video.mp4"
    video_source = 0 
    detect_traffic_light(video_source)