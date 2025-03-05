import cv2
from ultralytics import YOLO

# Constants
MODEL_PATH = 'yolov8s.pt'
CONFIDENCE_THRESHOLD = 0.4
CAMERA_INDEX = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_THICKNESS = 2
RECT_THICKNESS = 2


def get_colors(class_index):
    """
    Generate a unique color for each class based on its index.
    
    Args:
        class_index: The index of the class
        
    Returns:
        A tuple representing the BGR color
    """
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR format
    color_index = class_index % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    
    color = [
        (base_colors[color_index][i] + 
         increments[color_index][i] * (class_index // len(base_colors))) % 256 
        for i in range(3)
    ]
    
    return tuple(color)


def process_detection(frame, result):
    """
    Process detection results and draw bounding boxes on the frame.
    
    Args:
        frame: The video frame to process
        result: Detection result from YOLO model
        
    Returns:
        The processed frame with bounding boxes
    """
    classes_names = result.names
    
    for box in result.boxes:
        # Skip detections with low confidence
        if box.conf[0] < CONFIDENCE_THRESHOLD:
            continue
            
        # Get coordinates and convert to integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get class information
        cls_index = int(box.cls[0])
        class_name = classes_names[cls_index]
        confidence = box.conf[0]
        
        # Get color for this class
        color = get_colors(cls_index)
        
        # Draw rectangle around the object
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, RECT_THICKNESS)
        
        # Add label with class name and confidence
        label = f'{class_name} {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1), FONT, FONT_SCALE, color, FONT_THICKNESS)
    
    return frame


def main():
    """Main function to run object detection on video stream."""
    # Load the YOLO model
    model = YOLO(MODEL_PATH)
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    
    if not video_capture.isOpened():
        print("Error: Could not open video source")
        return
    
    try:
        while True:
            # Read frame from video
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                continue
            
            # Run object detection and tracking
            results = model.track(frame, stream=True)
            
            # Process each result
            for result in results:
                frame = process_detection(frame, result)
            
            # Display the processed frame
            cv2.imshow('Object Detection', frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up resources
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()