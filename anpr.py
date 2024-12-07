import cv2
import torch
import easyocr
import os
from pathlib import Path

def load_yolov9_model(weights_path="yolov9.pt"):
    """
    Load the YOLOv9 model for license plate detection.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
    model.conf = 0.4  # Confidence threshold
    model.iou = 0.5   # Intersection-over-Union threshold
    return model
import torch
import cv2
import os
from pathlib import Path
import easyocr

# Initialize YOLOv9 model
def load_yolo_model(weights_path="yolov9.pt"):
    """
    Load the YOLOv9 model from the specified weights file.
    Args:
        weights_path (str): Path to the YOLOv9 weights file.
    Returns:
        YOLOv9 model loaded with specified weights.
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
        print("YOLOv9 model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLOv9 model: {e}")
        raise

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def process_frame(frame, model):
    """
    Process a single frame using YOLOv9 and EasyOCR.
    Args:
        frame (np.ndarray): The input video frame.
        model: The YOLOv9 object detection model.
    Returns:
        processed_frame (np.ndarray): Frame with bounding boxes and text annotations.
        plates (list): List of detected number plate texts.
    """
    results = model(frame)  # YOLOv9 inference
    detections = results.xyxy[0]  # Detections in [x1, y1, x2, y2, conf, class]

    plates = []
    for *box, conf, cls in detections.tolist():
        if int(cls) == 0:  # Assuming '0' is the class for number plates
            x1, y1, x2, y2 = map(int, box)
            cropped_plate = frame[y1:y2, x1:x2]

            # OCR detection
            ocr_results = reader.readtext(cropped_plate)
            for (bbox, text, conf) in ocr_results:
                if conf > 0.5:  # Threshold for OCR confidence
                    plates.append(text)
                    # Annotate frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return frame, plates

def run(source, project, name, exist_ok=False, save_txt=True, nosave=False):
    """
    Run the ANPR pipeline on a video file.
    Args:
        source (str): Path to the input video file.
        project (str): Directory where outputs will be saved.
        name (str): Name for the output directory/file.
        exist_ok (bool): Allow overwriting of output files.
        save_txt (bool): Save detected plate numbers as a text file.
        nosave (bool): If True, skip saving the processed video.
    """
    # Load YOLOv9 model
    model = load_yolo_model()

    # Prepare output paths
    output_dir = Path(project) / name
    output_dir.mkdir(parents=True, exist_ok=exist_ok)
    output_video_path = output_dir / f"{name}.mp4"
    output_text_path = output_dir / f"{name}.txt"

    # Initialize video capture and writer
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    plates_set = set()  # To store unique plate numbers
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame, plates = process_frame(frame, model)
        plates_set.update(plates)

        # Write the processed frame to output video
        if not nosave:
            out.write(processed_frame)

    cap.release()
    out.release()

    # Save detected plate numbers to text file
    if save_txt:
        with open(output_text_path, "w") as f:
            for plate in plates_set:
                f.write(f"{plate}\n")

    print(f"Processing complete. Output saved at: {output_dir}")

def detect_license_plates(model, frame):
    """
    Use the YOLO model to detect license plates in a single frame.
    """
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2, confidence, class]
    return detections

def extract_text_from_plate(image, coordinates):
    """
    Extract text from the detected license plate using EasyOCR.
    """
    x1, y1, x2, y2 = [int(coord) for coord in coordinates]
    cropped_plate = image[y1:y2, x1:x2]

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_plate)

    if result:
        return result[0][1]  # Return the detected text
    return None

def process_video(source, output_path, model, output_text_path=None):
    """
    Process the input video to detect and extract license plate numbers.
    """
    # Open the video file
    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Optional: Open text file to save detected license plates
    if output_text_path:
        text_file = open(output_text_path, "w")

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        detections = detect_license_plates(model, frame)

        for detection in detections:
            x1, y1, x2, y2, confidence, cls = detection
            if int(cls) == 0:  # Class 0 for license plates
                text = extract_text_from_plate(frame, (x1, y1, x2, y2))
                if text:
                    # Draw rectangle and add text to the frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Save detected plate text
                    if output_text_path:
                        text_file.write(f"Frame {frame_number}: {text}\n")

        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    if output_text_path:
        text_file.close()

def run(source, project="output", name="processed", exist_ok=True, save_txt=True, nosave=False):
    """
    Main function to run the ANPR pipeline.
    """
    # Load YOLOv9 model
    weights_path = "C:/Users/hites/OneDrive/Desktop/CV-Hack/yolov9/runs/train/exp/weights/best.pt"  # Path to your YOLOv9 weights file
    model = load_yolov9_model(weights_path)

    # Create output directories
    project_path = Path(project)
    if not exist_ok and project_path.exists():
        raise FileExistsError(f"Project directory '{project_path}' already exists!")
    project_path.mkdir(parents=True, exist_ok=True)

    # Define output paths
    output_video_path = project_path / f"{name}.mp4"
    output_text_path = project_path / f"{name}.txt" if save_txt else None

    # Process the video
    process_video(source, str(output_video_path), model, str(output_text_path) if save_txt else None)

    # Cleanup
    if nosave:
        output_video_path.unlink()

if __name__ == "__main__":
    # Example usage
    run(
        source="input_video.mp4",
        project="output_videos",
        name="processed_video",
        exist_ok=True,
        save_txt=True,
        nosave=False
    )
