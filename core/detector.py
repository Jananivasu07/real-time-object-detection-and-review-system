from ultralytics import YOLO
import cv2

class Detector:
    """
    Advanced YOLOv8 wrapper with Object Tracking (ByteTrack).
    Tracking uses temporal context to improve accuracy and reduce flicker.
    """
    def __init__(self, model_path='data/yolov8s.pt', conf_threshold=0.35, iou_threshold=0.5):
        print(f"[INFO] Initializing Tracker with {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect_and_track(self, frame):
        """
        Runs YOLOv8 in tracking mode.
        Returns a list of detections with persistent tracking IDs.
        """
        # track() enables ByteTrack/BoT-SORT internally
        results = self.model.track(frame, 
                                   persist=True, 
                                   conf=self.conf_threshold, 
                                   iou=self.iou_threshold,
                                   imgsz=640,
                                   verbose=False)
        
        detections = []
        for result in results:
            if result.boxes and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, conf, cls in zip(boxes, ids, confs, clss):
                    label = result.names[cls]
                    detections.append({
                        'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'label': label,
                        'confidence': float(conf),
                        'track_id': int(track_id)
                    })
            elif result.boxes:
                # Fallback to normal detection if tracking failed for a frame
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = result.names[cls]
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'label': label,
                        'confidence': conf,
                        'track_id': -1
                    })
        
        return detections
