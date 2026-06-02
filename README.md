# Real-Time Object Detection and Best Object Review System
**A Project by Students of Saraswathy College of Engineering and Technology**

## Advanced Engineering Features (v2)
- **Zero-Latency Pipeline**: Multithreaded architecture decouples frame capture (60 FPS) from AI inference, ensuring a perfectly smooth UI even on lower-end hardware.
- **ByteTrack Integration**: Uses temporal object tracking to maintain identity across frames, significantly reducing misclassifications and flickering.
- **Voice Synthesis Engine**: Real-time audio announcements of the "Best Object" using `pyttsx3`.
- **High-Tech HUD**: Real-time telemetry, mission-style interface, and screen overlays.

## Features (Legacy)
- Real-time object detection using YOLOv8 (Nano model for speed).
- Custom Suitability Scoring: `(0.6 * confidence) + (0.4 * attribute_rating)`.
- Visual highlighting: Best object in **GREEN**, others in **BLUE**.
- Frame-by-frame logging to `detections_log.csv`.
- Periodic summary statistics in the terminal.

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Attributes:**
   Edit `attributes.json` to adjust the weight/importance of different object classes.

## Running the Application

### Standard Webcam Mode:
```bash
python main.py
```

### Video File Mode:
```bash
python main.py --source path/to/video.mp4
```

### Adjusting Confidence Threshold:
```bash
python main.py --conf 0.5
```

## Project Structure
The project is organized into a clean, modular structure:
```text
orbital-tyson/
├── core/               # AI & Logic Modules
│   ├── detector.py     # YOLO Tracking
│   ├── scorer.py       # Suitability Engine
│   ├── visualizer.py   # HUD Rendering
│   ├── logger.py       # Data Logging
│   ├── voice_engine.py # Voice Synthesis
│   └── face_analyzer.py# Human Intel Layer
├── data/               # Persistent Data & Models
│   ├── attributes.json # Configurable Weights
│   ├── yolov8s.pt      # Small Model (Optimized Accuracy)
│   ├── yolov8n.pt      # Nano Model (Legacy / Ultra-Speed)
│   └── detections_log.csv
├── faces/              # Identity Reference Images
├── templates/          # Web Interface (HTML)
├── static/             # UI Styling (CSS)
├── app.py              # Web Command Center (Flask)
├── main.py             # CLI Performance Entry
└── requirements.txt
```

## Controls
- Press **'q'** in the video window to quit.
- Press **'f'** to toggle **Fullscreen**.
- Press **'c'** to **Cycle** between available cameras (if multiple are detected).

---

## 🚀 How to Migrate to Another Laptop (Presentation Guide)
To move this project for your college demonstration, follow these simple steps on the new laptop:

1. **Transfer the Files**: 
   Copy the entire `orbital-tyson` folder (including `templates`, `static`, `faces`, and `.pt` model files) to the new laptop via Pen Drive or Cloud.

2. **Install Python**: 
   Ensure [Python 3.10+](https://www.python.org/downloads/) is installed. Check by running `python --version` in terminal.

3. **Install Dependencies**: 
   Open a terminal (PowerShell or CMD) inside the project folder and run:
   ```powershell
   pip install -r requirements.txt
   ```
   *Note: This will install Flask, Ultralytics, OpenCV, and other required AI libraries.*

4. **Verify Hardware**:
   Ensure the laptop has a working webcam. If using an external USB camera, it will usually be detected automatically on index `0` or `1`.

5. **Run the System**:
   ```powershell
   python app.py
   ```
   Then open **`http://127.0.0.1:5000`** in any browser.

6. **Tip for Face Detection**:
   If the new laptop doesn't have `face_recognition` installed (due to dlib complex setup), the project **will still run perfectly** in resilient mode, showing detections and logs on the web dashboard.
