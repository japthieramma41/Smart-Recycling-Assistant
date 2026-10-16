from pathlib import Path
from datetime import datetime
import json
import csv
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import plotly.express as px

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "waste_classifier.keras"
CLASS_PATH = APP_DIR / "models" / "class_names.json"
RECORDS_PATH = APP_DIR / "data" / "classification_records.csv"
IMAGE_SIZE = (224, 224)
LOW_CONFIDENCE_THRESHOLD = 0.55

GUIDANCE = {
    "plastic": ("Plastic / designated dry recyclables", "Empty and clean recyclable containers. Check local rules because not all plastics are accepted."),
    "paper": ("Paper recycling", "Keep paper clean and dry. Greasy or contaminated paper may not be accepted."),
    "cardboard": ("Cardboard / designated dry recyclables", "Flatten clean cardboard and follow local collection rules."),
    "glass": ("Designated glass collection", "Use the appropriate glass collection stream where available; handle broken glass carefully."),
    "metal": ("Metal recycling", "Empty containers and use the designated metal or dry-recyclables stream."),
    "trash": ("Check local residual-waste guidance", "This model category is broad. Check the item and local rules before disposal.")
}
# Simulated fill increment per confirmed item; these are assumptions, not measured volumes.
FILL_INCREMENT = {"plastic": 2.0, "paper": 1.5, "cardboard": 3.0, "glass": 2.0, "metal": 1.5, "trash": 2.0}

st.set_page_config(page_title="Smart Recycling Assistant", page_icon="♻️", layout="wide")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not CLASS_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

def ensure_data_file():
    RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not RECORDS_PATH.exists():
        with open(RECORDS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "predicted_category", "confidence", "confirmed", "source"])
    return RECORDS_PATH

def read_records():
    ensure_data_file()
    try:
        df = pd.read_csv(RECORDS_PATH)
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "predicted_category", "confidence", "confirmed", "source"])
        df["predicted_category"] = df["predicted_category"].astype(str).str.lower()
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "predicted_category", "confidence", "confirmed", "source"])

def add_record(category, confidence, source="uploaded image"):
    ensure_data_file()
    with open(RECORDS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(timespec="seconds"), category, round(float(confidence), 4), True, source])

def display_bin_metrics(df, classes):
    st.subheader("Digital Twin: Virtual Waste Bins")
    st.caption("Fill levels are simulated using assumed increments per confirmed item. They are not real sensor measurements.")
    cols = st.columns(3)
    counts = df["predicted_category"].value_counts().to_dict() if not df.empty else {}
    for idx, category in enumerate(classes):
        count = int(counts.get(category, 0))
        fill = min(100.0, count * FILL_INCREMENT.get(category, 2.0))
        with cols[idx % 3]:
            st.metric(f"{category.title()} Bin", f"{fill:.0f}% simulated fill", f"{count} confirmed items")
            st.progress(int(fill))

st.title("♻️ AI-Powered Smart Recycling Assistant")
st.write("Upload a waste image, review the AI prediction, and update a virtual waste-bin dashboard.")
st.info("Prototype note: AI predictions can be wrong. Confirm the category before adding it to the dashboard. Bin fill levels are simulated.")

model, class_names = load_model()
if model is None:
    st.error("The trained model was not found. Follow README.md to train the model and save files in the models/ folder.")
    st.stop()

tab_classify, tab_dashboard, tab_about = st.tabs(["Classify Waste", "Digital Twin Dashboard", "About"])

with tab_classify:
    left, right = st.columns([1, 1])
    with left:
        uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
    with right:
        if uploaded_file:
            if st.button("Classify Waste", type="primary"):
                image_resized = image.resize(IMAGE_SIZE)
                array = np.asarray(image_resized, dtype=np.float32)
                probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
                best_index = int(np.argmax(probabilities))
                st.session_state["last_prediction"] = {
                    "category": class_names[best_index],
                    "confidence": float(probabilities[best_index]),
                    "probabilities": {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                }

        prediction = st.session_state.get("last_prediction")
        if prediction:
            category = prediction["category"].lower()
            confidence = prediction["confidence"]
            st.subheader("Prediction result")
            st.metric("Predicted category", category.title())
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.warning("Low-confidence prediction. Check the image and category before confirming.")
            if category in GUIDANCE:
                bin_name, tip = GUIDANCE[category]
                st.write(f"**Suggested disposal stream:** {bin_name}")
                st.write(f"**Tip:** {tip}")
            st.caption("The model predicts one of its trained classes; it does not verify recyclability or local collection rules.")
            if st.button("Confirm classification and update virtual bin"):
                add_record(category, confidence)
                st.success("Classification recorded. Open the Digital Twin Dashboard to view the updated counts.")
                st.session_state.pop("last_prediction", None)
                st.rerun()

with tab_dashboard:
    records = read_records()
    if records.empty:
        st.write("No confirmed classifications yet. Use the Classify Waste tab to add the first item.")
    else:
        total = len(records)
        st.metric("Total confirmed classifications", total)
        counts_df = records["predicted_category"].value_counts().rename_axis("category").reset_index(name="items")
        fig = px.bar(counts_df, x="category", y="items", title="Confirmed Items by Category")
        st.plotly_chart(fig, use_container_width=True)
        display_bin_metrics(records, class_names)
        with st.expander("View recent records"):
            st.dataframe(records.tail(20).sort_values("timestamp", ascending=False), use_container_width=True)
        csv_bytes = records.to_csv(index=False).encode("utf-8")
        st.download_button("Download classification records (CSV)", csv_bytes, "classification_records.csv", "text/csv")
        if st.button("Clear all prototype records"):
            ensure_data_file()
            pd.DataFrame(columns=["timestamp", "predicted_category", "confidence", "confirmed", "source"]).to_csv(RECORDS_PATH, index=False)
            st.success("Records cleared.")
            st.rerun()

with tab_about:
    st.subheader("Project overview")
    st.write("This prototype combines image classification, disposal guidance, and a simulated Digital Twin dashboard.")
    st.write("Dataset: TrashNet — https://github.com/garythung/trashnet")
    st.write("Model: MobileNetV2 with transfer learning.")
    st.write("SDG alignment: SDG 12 (Responsible Consumption and Production) and SDG 11 (Sustainable Cities and Communities).")
    st.write("Future work: campus-specific image collection, physical bin sensors, improved validation, and multilingual guidance.")
