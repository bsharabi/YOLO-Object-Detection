# Object Detection with YOLO and OpenCV

This project demonstrates real-time object detection using the YOLO (You Only Look Once) model and OpenCV. The script captures video from a webcam, processes each frame to detect objects, and displays bounding boxes with class names and confidence scores.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/bsharabi/YOLO-Object-Detection.git
    cd YOLO-Object-Detection
    ```

2. **Create a Virtual Environment (Optional but recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script:
```bash
python ObjectDetection.py
```

## Requirements
- Python 3.x
- OpenCV
- Ultralytics YOLO
