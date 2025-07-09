import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Optional webcam: only import if running locally
try:
    from streamlit_webrtc import webrtc_streamer
    import av
    webcam_available = True
except ImportError:
    webcam_available = False

# ---------------- Page Config ----------------
st.set_page_config(page_title="YOLOv11 Multi-Task App", layout="centered")
st.title("🤖 Multi-Task Vision App")
st.markdown(
    "Use Ultralytics YOLOv11 models to perform real-time **Detection, Classification, Segmentation, Pose**, and **OBB** on images, videos, or webcam feeds."
)

# ---------------- Detect Hugging Face ----------------
on_spaces = "SPACE_ID" in os.environ or "HF_SPACE_ID" in os.environ

# ---------------- Sidebar ----------------
st.sidebar.title("🔧 Configuration")
task_type = st.sidebar.selectbox("📌 Select Task", ["Select", "Detection", "Classification", "Segmentation", "Pose", "OBB"])
model_size = st.sidebar.selectbox("📦 Select Model Size", ["Select", "YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"])

# Conditionally show webcam
input_type_options = ["Select", "Image", "Video"]
if not on_spaces and webcam_available:
    input_type_options.append("Webcam")
input_type = st.sidebar.selectbox("🎯 Select Input Type", input_type_options)

# ---------------- Safety Check ----------------
if task_type == "Select" or model_size == "Select" or input_type == "Select":
    st.warning("⚠️ Please select a task, model size, and input type from the sidebar.")
    st.stop()

# ---------------- Model Mapping ----------------
MODEL_PATHS = {
    "Detection": {f"YOLOv11{size}": f"yolo11{size}.pt" for size in ["n", "s", "m", "l", "x"]},
    "Classification": {f"YOLOv11{size}": f"yolo11{size}-cls.pt" for size in ["n", "s", "m", "l", "x"]},
    "Segmentation": {f"YOLOv11{size}": f"yolo11{size}-seg.pt" for size in ["n", "s", "m", "l", "x"]},
    "Pose": {f"YOLOv11{size}": f"yolo11{size}-pose.pt" for size in ["n", "s", "m", "l", "x"]},
    "OBB": {f"YOLOv11{size}": f"yolo11{size}-obb.pt" for size in ["n", "s", "m", "l", "x"]},
}
model_path = MODEL_PATHS[task_type][model_size]

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_path)

# ---------------- IMAGE ----------------
if input_type == "Image":
    image_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])
    if image_file:
        img = Image.open(image_file).convert("RGB")
        st.image(img, caption="Uploaded Image")
        img_np = np.array(img)[:, :, ::-1]
        results = model(img_np, verbose=False)[0]

        if task_type == "Classification":
            label = results.names[results.probs.top1]
            conf = results.probs.top1conf.item()
            st.success(f"✅ Predicted: **{label}** with confidence **{conf:.2f}**")
        else:
            output_img = results.plot()
            st.image(output_img, caption=f"{task_type} Result")

# ---------------- VIDEO ----------------
elif input_type == "Video":
    video_file = st.file_uploader("🎬 Upload a Video", type=["mp4", "avi", "mov"])
    if video_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(video_file.read())
        video_path = temp_video.name
        st.video(video_path)

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        frame_count = 0
        max_frames = 200

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > max_frames:
                break

            processed = cv2.resize(frame, (224, 224)) if task_type == "Classification" else frame
            results = model(processed, verbose=False)[0]

            if task_type == "Classification":
                label = results.names[results.probs.top1]
                conf = results.probs.top1conf.item()
                cv2.putText(processed, f"{label} ({conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                stframe.image(processed, channels="BGR")
            else:
                output_img = results.plot()
                stframe.image(output_img, channels="BGR")
            frame_count += 1

        cap.release()
        st.success("✅ Video processing completed.")

# ---------------- WEBCAM (Only If Local) ----------------
elif input_type == "Webcam" and webcam_available and not on_spaces:
    st.info("📷 Allow camera access in your browser to start real-time YOLO detection.")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        processed = cv2.resize(img, (224, 224)) if task_type == "Classification" else img
        try:
            results = model(processed, verbose=False)[0]
            if task_type == "Classification":
                label = results.names[results.probs.top1]
                conf = results.probs.top1conf.item()
                cv2.putText(processed, f"{label} ({conf:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return av.VideoFrame.from_ndarray(processed, format="bgr24")
            else:
                output_img = results.plot()
                return av.VideoFrame.from_ndarray(output_img, format="bgr24")
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="live",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif input_type == "Webcam" and on_spaces:
    st.error("🚫 Webcam is not supported on Hugging Face Spaces. Please run this app locally to use webcam input.")

