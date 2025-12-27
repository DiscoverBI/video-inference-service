# Vision Inference Service

**AGPL-3.0 Licensed Component**

Isolierter KI-Dienst für Personenerkennung und Pose-Analyse mit YOLOv8.

## Lizenz

Dieser Service ist unter AGPL-3.0 lizenziert, da er Ultralytics YOLO verwendet.

## Installation

```bash
pip install -r requirements.txt
```

## Start

```bash
python app.py
```

Der Service läuft auf `http://127.0.0.1:5001`

## API Endpunkte

### GET /health
Health Check

### POST /analyze
Frame-Analyse

**Request:**
```json
{
  "image": "base64_encoded_image",
  "camera_name": "cam1",
  "imgsz": 1280,
  "conf_threshold": 0.5
}
```

**Response:**
```json
{
  "persons": [{
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.92,
    "keypoints": [[x, y, conf], ...],
    "is_lying_down": false,
    "angle": 85.5,
    "position": {"x": 640, "y": 360}
  }],
  "timestamp": "2025-01-10T16:57:21",
  "camera_name": "cam1"
}
```

## Docker

```bash
docker build -t vision-inference-service .
docker run -p 5001:5001 vision-inference-service
```

## Austauschbarkeit

Dieser Service kann durch jede andere Implementierung ersetzt werden,
die dieselbe HTTP API bereitstellt.
