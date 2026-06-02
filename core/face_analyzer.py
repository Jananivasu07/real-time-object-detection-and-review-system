import os
import numpy as np

# Optional imports with fallbacks
try:
    import face_recognition
    import cvlib as cv
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
    print("[WARNING] Face recognition or CVLib not installed. Identity features disabled.")

class HumanAnalyzer:
    """
    Handles Face Recognition (Identity) and Gender Detection.
    Resilient to missing dependencies.
    """
    def __init__(self, faces_dir='faces'):
        self.faces_dir = faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        if HAS_LIBS:
            self.load_known_faces()
            print(f"[INFO] HumanAnalyzer initialized with {len(self.known_face_names)} identities.")
        else:
            print("[INFO] HumanAnalyzer running in STUB mode (missing deps).")

    def load_known_faces(self):
        if not HAS_LIBS: return
        import cv2
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            return

        for filename in os.listdir(self.faces_dir):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                path = os.path.join(self.faces_dir, filename)
                try:
                    image = face_recognition.load_image_file(path)
                    encoding = face_recognition.face_encodings(image)
                    if len(encoding) > 0:
                        self.known_face_encodings.append(encoding[0])
                        name = os.path.splitext(filename)[0].capitalize()
                        self.known_face_names.append(name)
                except Exception as e:
                    print(f"[ERROR] Could not load face {filename}: {e}")

    def analyze_person(self, person_crop):
        analysis = {
            'identity': "UNKNOWN",
            'gender': "UNKNOWN"
        }

        if not HAS_LIBS or person_crop is None or person_crop.size == 0:
            return analysis

        import cv2
        # 1. Identity Check
        try:
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_crop)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            analysis['identity'] = self.known_face_names[best_match_index]
                            break
        except Exception: pass

        # 2. Gender Detection
        try:
            face, label, confidence = cv.detect_face(person_crop)
            if label:
                gender_labels, confs = cv.detect_gender(person_crop)
                if gender_labels:
                    analysis['gender'] = gender_labels[0].upper()
        except Exception: pass

        return analysis
