"""
Vision Inference Service - AGPL-3.0 Licensed
Isolierter KI-Dienst für Personenerkennung und Pose-Analyse
Kommuniziert ausschließlich über HTTP REST API
"""
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import math
import logging
from datetime import datetime

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Globales Modell (wird beim Start geladen)
model = None

def load_model(model_path='yolov8n-pose.pt'):
    """Lädt das YOLO Pose Model"""
    global model
    try:
        model = YOLO(model_path)
        logging.info(f"✓ YOLO Model geladen: {model_path}")
        return True
    except Exception as e:
        logging.error(f"✗ Fehler beim Laden des Models: {e}")
        return False

def decode_image(base64_string):
    """Dekodiert Base64 String zu OpenCV Image"""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"Fehler beim Dekodieren des Bildes: {e}")
        return None

def is_person_lying_down(keypoints, bbox):
    """
    Verbesserte Man-Down-Erkennung basierend auf Körperposition
    """
    try:
        # Extrahiere Keypoints mit Confidence-Check
        nose = keypoints[0] if len(keypoints) > 0 and keypoints[0][2] > 0.5 else None
        left_shoulder = keypoints[5] if len(keypoints) > 5 and keypoints[5][2] > 0.5 else None
        right_shoulder = keypoints[6] if len(keypoints) > 6 and keypoints[6][2] > 0.5 else None
        left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.5 else None
        right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.5 else None
        
        # Berechne Durchschnittspositionen
        shoulder_y = None
        hip_y = None
        
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        elif left_shoulder is not None:
            shoulder_y = left_shoulder[1]
        elif right_shoulder is not None:
            shoulder_y = right_shoulder[1]
            
        if left_hip is not None and right_hip is not None:
            hip_y = (left_hip[1] + right_hip[1]) / 2
        elif left_hip is not None:
            hip_y = left_hip[1]
        elif right_hip is not None:
            hip_y = right_hip[1]
        
        # Faktor 1: Vertikaler Schulter-Hüft-Abstand
        vertical_distance_factor = 0
        if shoulder_y is not None and hip_y is not None:
            vertical_distance = abs(hip_y - shoulder_y)
            bbox_height = bbox[3] - bbox[1]
            
            if bbox_height > 0:
                relative_distance = vertical_distance / bbox_height
                # Stehend: >0.3, Liegend: <0.2
                vertical_distance_factor = 1 if relative_distance < 0.25 else 0
        
        # Faktor 2: Bounding Box Seitenverhältnis
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        
        aspect_ratio_factor = 0
        if bbox_height > 0:
            aspect_ratio = bbox_width / bbox_height
            # Breiter als hoch deutet auf liegend hin
            aspect_ratio_factor = 1 if aspect_ratio > 1.2 else 0
        
        # Entscheidung: Liegend wenn beide Faktoren zutreffen
        is_lying = (vertical_distance_factor + aspect_ratio_factor) >= 2
        
        return is_lying
        
    except Exception as e:
        logging.error(f"Fehler bei Man-Down-Erkennung: {e}")
        return False

def calculate_angle(keypoints):
    """Berechnet den Körperwinkel für Visualisierung"""
    try:
        left_shoulder = keypoints[5] if len(keypoints) > 5 and keypoints[5][2] > 0.5 else None
        right_shoulder = keypoints[6] if len(keypoints) > 6 and keypoints[6][2] > 0.5 else None
        left_hip = keypoints[11] if len(keypoints) > 11 and keypoints[11][2] > 0.5 else None
        right_hip = keypoints[12] if len(keypoints) > 12 and keypoints[12][2] > 0.5 else None
        
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None
            
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2, 
                  (left_hip[1] + right_hip[1]) / 2)
        
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        
        angle_rad = math.atan2(abs(dx), abs(dy))
        angle_deg = math.degrees(angle_rad)
        
        return round(angle_deg, 1)
    except:
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health Check Endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    """
    Hauptendpoint für Frame-Analyse
    
    Request Body:
    {
        "image": "base64_encoded_image",
        "camera_name": "cam1",
        "imgsz": 1280,
        "conf_threshold": 0.5
    }
    
    Response:
    {
        "persons": [{
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.92,
            "keypoints": [[x, y, conf], ...],
            "is_lying_down": false,
            "angle": 85.5,
            "position": {"x": 640, "y": 360}
        }],
        "timestamp": "2025-01-10T16:57:21"
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.get_json()
        
        # Validierung
        if 'image' not in data:
            return jsonify({'error': 'Missing image data'}), 400
        
        # Parameter
        camera_name = data.get('camera_name', 'unknown')
        imgsz = data.get('imgsz', 1280)
        conf_threshold = data.get('conf_threshold', 0.5)
        
        # Dekodiere Bild
        frame = decode_image(data['image'])
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # YOLO Inferenz
        results = model.track(
            frame,
            persist=True,
            conf=conf_threshold,
            iou=0.5,
            imgsz=imgsz,
            verbose=False
        )
        
        persons = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Keypoints (falls vorhanden)
                keypoints_data = None
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()
                
                for idx in range(len(boxes)):
                    bbox = boxes[idx].tolist()
                    confidence = float(confidences[idx])
                    
                    # Berechne Center Position
                    center_x = int((bbox[0] + bbox[2]) / 2)
                    center_y = int((bbox[1] + bbox[3]) / 2)
                    
                    person_data = {
                        'bbox': bbox,
                        'confidence': round(confidence, 2),
                        'position': {'x': center_x, 'y': center_y},
                        'keypoints': [],
                        'is_lying_down': False,
                        'angle': None
                    }
                    
                    # Keypoint-Analyse
                    if keypoints_data is not None and idx < len(keypoints_data):
                        kpts = keypoints_data[idx]
                        person_data['keypoints'] = kpts.tolist()
                        
                        # Man-Down-Erkennung
                        person_data['is_lying_down'] = is_person_lying_down(kpts, bbox)
                        
                        # Winkel berechnen
                        person_data['angle'] = calculate_angle(kpts)
                    
                    persons.append(person_data)
        
        return jsonify({
            'persons': persons,
            'timestamp': datetime.now().isoformat(),
            'camera_name': camera_name
        })
        
    except Exception as e:
        logging.error(f"Fehler bei Frame-Analyse: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Lade Modell beim Start
    if not load_model():
        logging.error("Konnte Modell nicht laden - Server wird nicht gestartet")
        exit(1)
    
    # Starte Server
    app.run(host='127.0.0.1', port=5001, debug=False)
