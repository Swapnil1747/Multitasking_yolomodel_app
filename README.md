# 🤖 YOLOv11 Multi-Task Vision App

## Overview
This project is a Streamlit-based web application that leverages Ultralytics YOLOv11 models to perform real-time multi-task vision processing. It supports a variety of computer vision tasks including **Detection, Classification, Segmentation, Pose Estimation**, and **Oriented Bounding Box (OBB)** detection on images, videos, and live webcam feeds.

## Features
- Multi-task support: Detection, Classification, Segmentation, Pose, and OBB
- Input types: Image, Video, and Webcam (local only)
- Model selection among YOLOv11 model sizes: n, s, m, l, x
- Real-time processing with live webcam streaming (using streamlit-webrtc)
- Upload and process images and videos with visualized results
- Snapshot capture and download support
- Performance optimizations with model caching

## Tech Stack
- Python
- Streamlit
- Ultralytics YOLOv11
- OpenCV
- Pillow (PIL)
- NumPy
- streamlit-webrtc (for webcam streaming)
- av (for video frame processing)

## Installation & Setup

### System Dependencies
Before installing Python dependencies, ensure your system has the necessary packages installed. For Ubuntu/Debian, you can run:
```
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
```

### Python Environment Setup
1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   - On Windows:
     ```
     python -m venv myenv
     myenv\Scripts\activate
     ```
   - On Linux/Mac:
     ```
     python3 -m venv myenv
     source myenv/bin/activate
     ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```
   streamlit run one/app.py
   ```
2. Use the sidebar to configure:
   - Select the task: Detection, Classification, Segmentation, Pose, or OBB
   - Select the model size: YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l, or YOLOv11x
   - Select the input type: Image, Video, or Webcam (webcam only available locally)

3. Upload images or videos, or start the webcam stream to see real-time results.

## Testing & Validation
- Test with sample images and videos by uploading them through the app.
- Use the webcam input for live detection if running locally.
- Verify results visually in the app interface.

## Deployment
- The app can be deployed on any platform supporting Python and Streamlit.
- For cloud deployment, consider platforms like Heroku, AWS, or Hugging Face Spaces (note: webcam input is disabled on Hugging Face Spaces).

## Future Work
- Add support for custom trained YOLOv11 models.
- Enhance UI/UX with more interactive controls and real-time performance metrics.
- Extend support for additional computer vision tasks and models.

## License & Attribution
- YOLOv11 models and code are used under their respective licenses.
- This project is intended for educational and research purposes.

## Contact
For issues, feature requests, or contributions, please open an issue or pull request in the repository.
