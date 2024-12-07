# Automatic Number Plate Recognition (ANPR)
---
Here's a video exlaination of **Automatic Number Plate Recognition (ANPR)** project attached:

[Explaination video]([https://drive.google.com/drive/folders/1Z9BU38if3ipHCBS-l6EpwVP0bo9bdCEB?usp=sharing])


This project implements an **Automatic Number Plate Recognition (ANPR)** system for smart city traffic management. It uses YOLOv9 for license plate detection and EasyOCR for text recognition. The solution is wrapped in a Streamlit application to allow users to process input videos and extract number plates.

## Features
- Detects number plates in video inputs using YOLOv9.
- Extracts text from detected number plates using EasyOCR.
- Outputs a processed video highlighting detected plates and displaying extracted text.
- Streamlit-based user interface for seamless interaction.

---

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.8 or later
- Streamlit
- OpenCV
- YOLOv9 and its dependencies
- EasyOCR
- Other dependencies as specified in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd anpr_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLOv9 pre-trained weights:
   - Place the weights file in the `models/` directory (or the path specified in the code).

---

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Upload a video for processing through the web interface.

3. View and download the output video with detected number plates and extracted text.

---

## File Structure
- `app.py`: Streamlit application script.
- `models/`: Directory for YOLOv9 weights.
- `utils/`: Helper functions for video processing and OCR.
- `requirements.txt`: List of dependencies.
- `example_videos/`: Sample input videos for testing.
- `output/`: Directory where processed videos are saved.

---

## Technologies Used
- **YOLOv9**: For fast and accurate number plate detection.
- **EasyOCR**: For robust text extraction from images.
- **OpenCV**: For video processing.
- **Streamlit**: For building an interactive user interface.

---

## Acknowledgements
- [YOLOv9](https://github.com/ultralytics/yolov9)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Streamlit](https://streamlit.io/)

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you need further customizations based on the exact notebook content, feel free to share specific details!
