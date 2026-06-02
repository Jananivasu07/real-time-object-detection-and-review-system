from flask import Flask, render_template, Response, jsonify
import cv2
import time
import json
import threading
import queue
from main import ThreadedVideoCapture
from core.detector import Detector
from core.scorer import Scorer
from core.visualizer import Visualizer
from core.voice_engine import VoiceEngine
from core.face_analyzer import HumanAnalyzer
from core.logger import Logger

app = Flask(__name__)

# Global Components
detector = Detector()
scorer = Scorer()
visualizer = Visualizer()
voice = VoiceEngine()
human_intel = HumanAnalyzer()
logger = Logger()

# Streaming state
video_stream = None
log_queue = queue.Queue(maxsize=50)

def generate_frames():
    global video_stream
    if video_stream is None:
        video_stream = ThreadedVideoCapture(0).start()

    frame_count = 0
    while True:
        frame = video_stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        start_time = time.time()

        # Detection & Tracking
        detections = detector.detect_and_track(frame)
        
        # Human Intelligence Analysis
        current_logs = []
        for det in detections:
            if det['label'] == 'person':
                x1, y1, x2, y2 = det['bbox']
                h, w = frame.shape[:2]
                person_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                # Perform Deep Analysis
                det['human_data'] = human_intel.analyze_person(person_crop)
                
                name = det['human_data']['identity']
                if name != "UNKNOWN":
                    voice.announce(f"{name} identified")
                
                log_msg = f"PEOPLE: {name} ({det['human_data']['gender']})"
                current_logs.append(log_msg)

        # Scoring
        best_detection = scorer.get_best_object(detections)
        if best_detection:
            logger.log_frame(frame_count, best_detection)
            label = best_detection['label']
            if label != 'person' or best_detection.get('human_data', {}).get('identity') == "UNKNOWN":
                voice.announce(label)
            
            log_msg = f"BEST: {label.upper()} - Score: {best_detection.get('suitability_score', 0):.2f}"
            current_logs.append(log_msg)

        # Push to log queue for UI
        for msg in current_logs:
            if log_queue.full():
                log_queue.get()
            log_queue.put({"timestamp": time.strftime("%H:%M:%S"), "message": msg})

        # Rendering
        fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 30
        output_frame = visualizer.draw_detections(frame, detections, best_detection, fps)

        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_logs')
def get_logs():
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return jsonify(logs)

if __name__ == '__main__':
    print("[WEB] Launching Real-Time Object Detection and Best Object Review System")
    print("[WEB] Saraswathy College of Engineering and Technology - Final Year Project")
    print("[WEB] Local Dashboard: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
