import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

def test_deepsort():
    # Initialize tracker
    tracker = DeepSort(max_age=50)
    
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create dummy detections
    detections = [
        [[100, 100, 50, 50], 0.9, 0],  # [bbox, confidence, class_id]
        [[200, 200, 50, 50], 0.8, 1]
    ]
    
    # Test update_tracks
    try:
        tracks = tracker.update_tracks(detections, frame=frame)
        print("DeepSort test successful!")
        print(f"Number of tracks: {len(tracks)}")
        return True
    except Exception as e:
        print(f"DeepSort test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_deepsort() 