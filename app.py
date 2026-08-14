import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
diabetes_model = joblib.load("diabetes_model.pkl")

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="Diabetic Food AI",
    page_icon="🥗",
    layout="centered"
)

st.title("🥗 Diabetic Food AI")
st.write("Upload a food image to predict the food category.")

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "food_model.keras"

@st.cache_resource
def load_model():
    custom_objects = {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input
    }

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )

model = load_model()
st.subheader("🩺 Diabetes Prediction")

glucose = st.number_input("Glucose", min_value=0.0, value=180.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=90.0)
bmi = st.number_input("BMI", min_value=0.0, value=34.0)
insulin = st.number_input("Insulin", min_value=0.0, value=200.0)
pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.8
)
age = st.number_input("Age", min_value=1, max_value=120, value=55)

if st.button("Check Diabetes Status"):

    patient = pd.DataFrame(
        [[age, glucose, blood_pressure, bmi, insulin, pedigree]],
        columns=[
            "Age",
            "Glucose",
            "BloodPressure",
            "BMI",
            "Insulin",
            "DiabetesPedigreeFunction"
        ]
    )

    diabetes_result = diabetes_model.predict(patient)[0]

    if diabetes_result == 1:
        st.error("🔴 Diabetes Prediction: Diabetic")
    else:
        st.success("🟢 Diabetes Prediction: Not Diabetic")

# -----------------------------
# Food classes
# -----------------------------
class_names = [
    "Healthy",
    "Unhealthy"
]

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Food Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Food Image",
        use_container_width=True
    )

    # Resize image
    img = image.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # MobileNetV2 preprocessing
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_class = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0])) * 100

    result = class_names[predicted_class]

    st.subheader("🔍 Prediction")

    if result == "Healthy":
        st.success(f"🟢 {result}")
    else:
        st.error(f"🔴 {result}")

    st.write(f"Confidence: **{confidence:.2f}%**")
