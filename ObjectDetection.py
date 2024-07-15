import cv2
import math
from ultralytics import YOLO

class ObjectDetection:
    """
    A class to handle object detection using YOLO and OpenCV.
    
    Attributes:
        model_path (str): Path to the YOLO model weights.
        class_names (list): List of object class names.
    """
    
    def __init__(self, model_path):
        """
        Initializes the ObjectDetection class with the given model path.
        
        Args:
            model_path (str): Path to the YOLO model weights.
        """
        self.model = YOLO(model_path)
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]
    
    def start_webcam(self):
        """
        Starts the webcam and performs object detection in real-time.
        """
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            success, img = cap.read()
            results = self.model(img, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
                    confidence = math.ceil((box.conf[0] * 100)) / 100  # confidence score
                    cls = int(box.cls[0])  # class id

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Display class name and confidence score
                    label = f"{self.class_names[cls]}: {confidence}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection("yolo-Weights/yolov8n.pt")
    detector.start_webcam()
