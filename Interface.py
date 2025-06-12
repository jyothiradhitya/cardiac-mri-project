# streamlit_app.py (Add future risk line graph for healthy vs predicted)
import streamlit as st
import torch
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from Model import UNet

@st.cache_resource
def load_model(model_path, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, 1))
    return torch.tensor(image, dtype=torch.float32)

def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output = (output > 0.5).float()
        return output.cpu().numpy()[0, 0]

def overlay_mask_on_image(image, mask):
    mask = cv2.resize(mask, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
    mask_colored = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    mask_colored[mask > 0.5] = [255, 0, 0]
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image_color, 0.7, mask_colored, 0.3, 0)

def highlight_chambers(mask):
    mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    overlay[mask == 1] = [0, 255, 255]  # RV - Cyan
    overlay[mask == 3] = [255, 255, 0]  # LV - Yellow
    return overlay

def compute_area(mask):
    return np.sum(mask > 0.5)

def diagnose_motion(area):
    variation = max(area) - min(area)
    if variation < 500:
        return "⚠️ Test case shows possible Cardiomyopathy (low motion variation)"
    return "✅ Normal motion detected"

def predict_future_risk(area):
    variation = max(area) - min(area)
    base_risk = 0.7 if variation < 500 else 0.2
    one_month = base_risk + 0.1
    three_months = base_risk + 0.15
    six_months = base_risk + 0.2
    year = min(base_risk + 0.3, 0.95)
    return one_month, three_months, six_months, year

def analyze_sequence(model, device, folder_path, max_frames=15):
    if not os.path.exists(folder_path):
        st.error(f"❌ Folder not found: {folder_path}")
        return [], [], [], []
    frames = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])[:max_frames]
    images, overlays, areas, chamber_views = [], [], [], []
    for fname in frames:
        img_path = os.path.join(folder_path, fname)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        input_tensor = preprocess_image(image)
        mask = predict_mask(model, input_tensor, device)
        overlay = overlay_mask_on_image(image, mask)
        highlight = highlight_chambers(mask)
        area = compute_area(mask)
        images.append(image)
        overlays.append(overlay)
        chamber_views.append(highlight)
        areas.append(area)
    return images, overlays, areas, chamber_views

def play_timelapse(images, overlays, label, speed=0.5):
    st.subheader(f"🎞️ {label} - Timelapse")
    col1, col2 = st.columns(2)
    frame1 = col1.empty()
    frame2 = col2.empty()
    for i in range(len(images)):
        frame1.image(images[i], caption=f"{label} Frame {i+1}", width=256)
        frame2.image(overlays[i], caption="Predicted Overlay", width=256)
        time.sleep(speed)

def plot_area_graph(healthy, diseased, test):
    st.subheader("📊 Area Progression Comparison")
    fig, ax = plt.subplots()
    ax.plot(healthy, marker='o', label="Healthy")
    ax.plot(diseased, marker='x', label="Diseased")
    ax.plot(test, marker='^', label="Test Case")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Segmentation Area (px)")
    ax.set_title("Area Trends: Test vs Healthy vs Diseased")
    ax.legend()
    st.pyplot(fig)

def plot_risk_forecast(risks):
    st.subheader("📈 Cardiomyopathy Risk Forecast")
    labels = ["1 Month", "3 Months", "6 Months", "1 Year"]
    values = [int(r * 100) for r in risks]
    fig, ax = plt.subplots()
    ax.plot(labels, values, marker='o', linestyle='-', color='crimson')
    ax.set_ylim(0, 100)
    ax.set_ylabel("Risk (%)")
    ax.set_title("Projected Cardiomyopathy Risk Over Time")
    st.pyplot(fig)

# --- Streamlit Layout ---
st.set_page_config(layout="wide")
st.title("🫀 Cardiac MRI Timelapse Analysis: Test case ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("unet_epoch_5.pth", device)

# Editable paths
st.sidebar.header("📂 Input Folder Paths")
def_healthy = r"C:\Users\sjyot\Downloads\Deep Learning-Based Cardiac MRI Timelapse Analysis\healthy\images"
def_diseased = r"C:\Users\sjyot\Downloads\Deep Learning-Based Cardiac MRI Timelapse Analysis\unhealthy\images"
def_test = r"C:\Users\sjyot\Downloads\Deep Learning-Based Cardiac MRI Timelapse Analysis\dataset\test\images"

healthy_folder = st.sidebar.text_input("Healthy Patient Folder", value=def_healthy)
diseased_folder = st.sidebar.text_input("Diseased Patient Folder", value=def_diseased)
test_folder = st.sidebar.text_input("Test Case Folder", value=def_test)
speed = st.sidebar.slider("Playback Speed", 0.1, 2.0, 0.5, step=0.1)

run_analysis = st.sidebar.button("▶️ Run Full Analysis")
run_visuals = st.sidebar.button("🔎 Visualise Progression & Highlight Chambers")

if run_analysis:
    if all([os.path.exists(healthy_folder), os.path.exists(diseased_folder), os.path.exists(test_folder)]):
        with st.spinner("Processing healthy patient..."):
            h_imgs, h_ovr, h_area, _ = analyze_sequence(model, device, healthy_folder)
        with st.spinner("Processing diseased patient..."):
            d_imgs, d_ovr, d_area, _ = analyze_sequence(model, device, diseased_folder)
        with st.spinner("Processing test case..."):
            t_imgs, t_ovr, t_area, t_highlight = analyze_sequence(model, device, test_folder)

        st.success("✅ All sequences processed successfully")

        col1, col2 = st.columns(2)
        with col1:
            play_timelapse(h_imgs, h_ovr, "Healthy Reference (from dataset)", speed)
        with col2:
            play_timelapse(d_imgs, d_ovr, "Unhealthy Reference (from dataset)", speed)

        st.header("🧪 Test Case Timelapse")
        play_timelapse(t_imgs, t_ovr, "Test Case", speed)

        st.subheader("📋 Diagnosis Result")
        st.markdown(diagnose_motion(t_area))

        plot_area_graph(h_area, d_area, t_area)

        df = pd.DataFrame({
            "Frame": list(range(1, len(t_area)+1)),
            "Healthy": h_area[:len(t_area)],
            "Diseased": d_area[:len(t_area)],
            "Test Case": t_area
        })
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), "comparison.csv", "text/csv")

        st.session_state.test_highlight = t_highlight
        st.session_state.test_area = t_area

if run_visuals and "test_highlight" in st.session_state:
    st.subheader("🫀 Left & Right Ventricle Highlight")
    for i, img in enumerate(st.session_state.test_highlight):
        st.image(img, caption=f"Test Frame {i+1} LV/RV", width=256)

    risks = predict_future_risk(st.session_state.test_area)
    st.subheader("📈 Predicted Risk of Cardiomyopathy")
    st.markdown(f"**In 1 Month:** {int(risks[0]*100)}% chance")
    st.markdown(f"**In 3 Months:** {int(risks[1]*100)}% chance")
    st.markdown(f"**In 6 Months:** {int(risks[2]*100)}% chance")
    st.markdown(f"**In 1 Year:** {int(risks[3]*100)}% chance")
    plot_risk_forecast(risks)
