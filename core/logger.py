import csv
import os
from datetime import datetime

class Logger:
    """
    CSV/JSON logging module for frame-by-frame analysis results.
    """
    def __init__(self, log_path='data/detections_log.csv'):
        """
        Initialize the logger and ensure headers are written to the CSV.
        """
        self.log_path = log_path
        self.headers = ['timestamp', 'frame_id', 'best_label', 'confidence', 'suitability_score']
        
        # Write headers if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log_frame(self, frame_id, best_detection):
        """
        Log the best object data of the current frame to the CSV file.
        """
        if best_detection is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = [
            timestamp,
            frame_id,
            best_detection['label'],
            f"{best_detection['confidence']:.4f}",
            f"{best_detection['suitability_score']:.4f}"
        ]
        
        try:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        except Exception as e:
            print(f"Error logging frame: {e}")
