import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# ---------------- Page Config ----------------
st.set_page_config(page_title="YOLOv11 Multi-Task App", layout="centered")
st.title("🤖 Multi-Task Vision App")
st.markdown(
    "Use Ultralytics YOLOv11 models to perform real-time **Detection, Classification, Segmentation, Pose**, and **OBB** on images, videos, or webcam feeds."
)

# ---------------- Detect Hosting ----------------
on_spaces = "SPACE_ID" in os.environ or "HF_SPACE_ID" in os.environ
on_render = os.environ.get("RENDER", "false") == "true"

# ---------------- Sidebar ----------------
st.sidebar.title("🔧 Configuration")
task_type = st.sidebar.selectbox("📌 Select Task", ["Select", "Detection", "Classification", "Segmentation", "Pose", "OBB"])
model_size = st.sidebar.selectbox("📦 Select Model Size", ["Select", "YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"])
input_type = st.sidebar.selectbox("🎯 Select Input Type", ["Select", "Image", "Video", "Webcam"])

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

# ---------------- WEBCAM (Streamlit-native) ----------------
elif input_type == "Webcam":
    st.info("📸 Use the button below to capture an image from your webcam.")

    img_file_buffer = st.camera_input("Capture Photo")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer).convert("RGB")
        st.image(img, caption="Captured Image")

        img_np = np.array(img)[:, :, ::-1]
        results = model(img_np, verbose=False)[0]

        if task_type == "Classification":
            label = results.names[results.probs.top1]
            conf = results.probs.top1conf.item()
            st.success(f"✅ Predicted: **{label}** with confidence **{conf:.2f}**")
        else:
            output_img = results.plot()
            st.image(output_img, caption=f"{task_type} Result")
