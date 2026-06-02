import cv2
import threading
import time
import queue
import argparse
import os
from collections import Counter
from tabulate import tabulate

from core.detector import Detector
from core.scorer import Scorer
from core.visualizer import Visualizer
from core.logger import Logger
from core.voice_engine import VoiceEngine
from core.face_analyzer import HumanAnalyzer

class ThreadedVideoCapture:
    def __init__(self, src, width=1280, height=720):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.q = queue.Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                return
            if not self.q.empty():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)

    def read(self):
        return self.q.get() if not self.q.empty() else None

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    parser = argparse.ArgumentParser(description="SCET AI DETECTION SUITE [V3]")
    parser.add_argument("--source", type=str, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--model", type=str, default="data/yolov8s.pt")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    # 1. Initialize High-Performance Components
    detector = Detector(model_path=args.model, conf_threshold=args.conf)
    scorer = Scorer()
    visualizer = Visualizer()
    logger = Logger()
    voice = VoiceEngine()
    human_intel = HumanAnalyzer()

    # 2. Start Video Stream
    src = int(args.source) if str(args.source).isdigit() else args.source
    video_stream = ThreadedVideoCapture(src).start()

    window_name = "SCET AI DETECTION SUITE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("[SYSTEM] HUMAN INTELLIGENCE LAYER ENGAGED")
    
    frame_count = 0
    
    while True:
        frame = video_stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        start_time = time.time()
        frame_count += 1

        # Phase 1: Object Detection & Tracking
        detections = detector.detect_and_track(frame)
        
        # Phase 2: Human Intelligence (Deep Analysis for Persons)
        for det in detections:
            if det['label'] == 'person':
                x1, y1, x2, y2 = det['bbox']
                h, w = frame.shape[:2]
                person_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                det['human_data'] = human_intel.analyze_person(person_crop)
                
                name = det['human_data']['identity']
                if name != "UNKNOWN":
                    voice.announce(f"{name} identified")

        # Phase 3: Scoring
        best_detection = scorer.get_best_object(detections)

        if best_detection:
            logger.log_frame(frame_count, best_detection)
            label = best_detection['label']
            if label != 'person' or (best_detection.get('human_data', {}).get('identity') == "UNKNOWN"):
                voice.announce(label)

        # Phase 4: HUD Rendering
        fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 30
        output_frame = visualizer.draw_detections(frame, detections, best_detection, fps)
        
        cv2.imshow(window_name, output_frame)

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('f'):
            is_full = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                                cv2.WINDOW_NORMAL if is_full == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)

    video_stream.stop()
    cv2.destroyAllWindows()
    print("[SYSTEM] SHUTDOWN COMPLETE")

if __name__ == "__main__":
    main()
