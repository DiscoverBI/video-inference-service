# Vision Inference Service - Deployment Guide

## Docker Deployment

### Build Image

```bash
docker build -t vision-service:latest .
```

### Run Container

```bash
docker run -d \
  --name vision-service \
  -p 5001:5001 \
  --restart unless-stopped \
  vision-service:latest
```

### Mit GPU-Unterstützung (NVIDIA)

```bash
docker run -d \
  --name vision-service \
  --gpus all \
  -p 5001:5001 \
  --restart unless-stopped \
  vision-service:latest
```

## Systemd Service (Linux)

Erstellen Sie `/etc/systemd/system/vision-service.service`:

```ini
[Unit]
Description=Vision Inference Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/vision-inference-service
ExecStart=/usr/bin/python3 /opt/vision-inference-service/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Aktivieren und starten:

```bash
sudo systemctl enable vision-service
sudo systemctl start vision-service
sudo systemctl status vision-service
```

## Windows Service

Verwenden Sie NSSM (Non-Sucking Service Manager):

```cmd
nssm install VisionService "C:\Python311\python.exe" "C:\vision-inference-service\app.py"
nssm set VisionService AppDirectory "C:\vision-inference-service"
nssm set VisionService DisplayName "Vision Service"
nssm set VisionService Description "AGPL-3.0 Vision Inference Service"
nssm set VisionService Start SERVICE_AUTO_START
nssm start VisionService
```

## Performance-Optimierung

### CPU-Modus
Standardmäßig läuft YOLO auf CPU. Für bessere Performance:

```python
# In app.py
model = YOLO(model_path)
model.to('cpu')  # Explizit CPU
```

### GPU-Modus (NVIDIA CUDA)
```python
# In app.py
model = YOLO(model_path)
model.to('cuda')  # NVIDIA GPU
```

### Modell-Größe anpassen

Kleineres Modell (schneller, weniger genau):
```python
model = YOLO('yolov8n-pose.pt')  # Nano
```

Größeres Modell (langsamer, genauer):
```python
model = YOLO('yolov8m-pose.pt')  # Medium
model = YOLO('yolov8l-pose.pt')  # Large
```

## Monitoring

### Health Check
```bash
curl http://localhost:5001/health
```

### Logs
```bash
# Docker
docker logs -f vision-service

# Systemd
journalctl -u vision-service -f
```

## Troubleshooting

### Service startet nicht
```bash
# Prüfe Python-Version
python --version  # Muss >= 3.8 sein

# Prüfe Dependencies
pip install -r requirements.txt

# Teste manuell
python app.py
```

### Langsame Inferenz
- GPU verwenden statt CPU
- Kleineres Modell verwenden
- `imgsz` Parameter reduzieren
- Mehr RAM/VRAM zuweisen

### Hoher Speicherverbrauch
- `imgsz` Parameter reduzieren
- Batch-Processing deaktivieren
- Modell auf CPU statt GPU
```

```py file="test_person_detection.py" isDeleted="true"
...deleted...
