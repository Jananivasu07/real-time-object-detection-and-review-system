import cv2
import numpy as np
from datetime import datetime

class Visualizer:
    """
    Premium High-Tech HUD visualizer with Human Intelligence overlays.
    """
    def __init__(self):
        self.neon_green = (57, 255, 20)
        self.neon_blue = (0, 255, 255)
        self.alert_red = (0, 0, 255)
        self.gold = (0, 215, 255)
        self.text_white = (255, 255, 255)
        self.header_bg = (10, 10, 10)
        
    def draw_corner_rect(self, img, pt1, pt2, color, thickness, corner_len):
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def draw_detections(self, frame, detections, best_detection, fps):
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. Global HUD Header
        cv2.rectangle(frame, (0, 0), (w, 45), self.header_bg, -1)
        cv2.line(frame, (0, 45), (w, 45), self.neon_blue, 1)
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, "SCET AI DETECTION SUITE", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.neon_blue, 1)
        cv2.putText(frame, f"CORE FPS: {fps:.1f}", (w - 180, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.alert_red, 1)

        # 2. Draw Detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            track_id = det.get('track_id', -1)
            
            is_best = (best_detection is not None and 
                       det['bbox'] == best_detection['bbox'] and 
                       det['label'] == best_detection['label'])
            
            color = self.neon_green if is_best else self.neon_blue
            thickness = 2
            
            # Corner Brackets for everyone
            self.draw_corner_rect(frame, (x1, y1), (x2, y2), color, thickness, 20)

            # Metadata Display
            meta_y = y1 - 10
            
            # Label
            display_text = f"{label.upper()} [ID:{track_id}]"
            cv2.putText(frame, display_text, (x1, meta_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Human Intelligence Data (if present)
            if 'human_data' in det:
                identity = det['human_data']['identity']
                gender = det['human_data']['gender']
                
                # Highlight Identity
                id_color = self.gold if identity != "UNKNOWN" else self.text_white
                cv2.putText(frame, f"USER: {identity}", (x1, y2 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, id_color, 1)
                cv2.putText(frame, f"GENDER: {gender}", (x1, y2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_white, 1)

            if is_best:
                # Top Best Badge
                score = det.get('suitability_score', 0)
                badge_text = f"TARGET LOCKED | SCORE: {score:.2f}"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + tw + 10, y1 - 10), self.neon_green, -1)
                cv2.putText(frame, badge_text, (x1 + 5, y1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 3. Aesthetics
        for y in range(45, h, 15):
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.03, frame, 0.97, 0, frame)

        return frame
