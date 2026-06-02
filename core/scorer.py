import json
import os

class Scorer:
    """
    Scoring engine logic to rank detected objects based on confidence and attributes.
    """
    def __init__(self, attributes_path='data/attributes.json'):
        """
        Initialize the scorer by loading attributes from attributes.json.
        """
        self.attributes = {}
        if os.path.exists(attributes_path):
            try:
                with open(attributes_path, 'r') as f:
                    self.attributes = json.load(f)
            except Exception as e:
                print(f"Error loading attributes: {e}")
        
        self.default_rating = self.attributes.get('default', 0.5)

    def compute_score(self, label, confidence):
        """
        Compute the Suitability Score for a detection.
        Formula: Score = (0.6 * confidence) + (0.4 * attribute_rating)
        """
        attribute_rating = self.attributes.get(label, self.default_rating)
        score = (0.6 * confidence) + (0.4 * attribute_rating)
        return score

    def get_best_object(self, detections):
        """
        Rank todos detected objects and return the one with the highest Suitability Score.
        detections: list of dicts with 'label', 'confidence', 'bbox', etc.
        """
        if not detections:
            return None
        
        for det in detections:
            det['suitability_score'] = self.compute_score(det['label'], det['confidence'])
        
        # Sort by score in descending order
        detections.sort(key=lambda x: x['suitability_score'], reverse=True)
        return detections[0]
