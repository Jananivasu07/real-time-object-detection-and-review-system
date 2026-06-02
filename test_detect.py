import cv2
import os
from detector import Detector
from scorer import Scorer
from visualizer import Visualizer

def test_on_image(image_path, output_path):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # Initialize modules
    detector = Detector(conf_threshold=0.3) # Lower threshold for test image
    scorer = Scorer()
    visualizer = Visualizer()

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    # 1. Detect objects
    detections = detector.detect(frame)
    print(f"Detected {len(detections)} objects.")

    # 2. Score objects and find the best one
    best_detection = scorer.get_best_object(detections)

    # 3. Visualize
    # We'll pass a fixed FPS of 0 for static image
    output_frame = visualizer.draw_detections(frame, detections, best_detection, 0.0)

    # 4. Save result
    cv2.imwrite(output_path, output_frame)
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    IMAGE_PATH = r"C:\Users\velav\.gemini\antigravity\brain\a1246d7d-d5ae-42ff-97c5-f3ba5e6558dc\test_scene_for_detection_1773072290000_1773071717451.png"
    OUTPUT_PATH = r"c:\Users\velav\.gemini\antigravity\playground\orbital-tyson\actual_output.jpg"
    test_on_image(IMAGE_PATH, OUTPUT_PATH)
