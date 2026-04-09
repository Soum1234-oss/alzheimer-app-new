import streamlit as st
'''
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from PIL import Image

IMG_SIZE = 224
classes = [
    "Non Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_models():
    svm = joblib.load("svm_local.pkl")
    pca = joblib.load("pca_local_zlib.pkl")

    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))
    vgg = Model(inputs=base_model.input, outputs=base_model.output)

    return svm, pca, vgg


# ---------------- VALIDATION FUNCTIONS ---------------- #

def is_grayscale(image, threshold=25):
    if len(image.shape) < 3:
        return True

    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    diff = np.mean(np.abs(r - g) + np.abs(r - b) + np.abs(g - b))

    return diff < threshold



    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Smooth image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 30, 120)

    edge_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # Relaxed range to avoid rejecting real MRI
    return 0.005 < edge_density < 0.3


# ---------------- UI ---------------- #

st.title("🧠 Alzheimer's Detection App (Validated MRI Only)")
st.write("Upload a brain MRI scan for dementia stage prediction.")

uploaded_file = st.file_uploader(
    "Choose MRI image...", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    # ---------------- VALIDATION ---------------- #

    # HARD CHECK (must pass)
    if not is_grayscale(img):
        st.error("❌ Invalid Image: MRI scans must be grayscale.")
        st.stop()

    # SOFT CHECK (warning only)
    if not has_brain_structure(img):
        st.warning("⚠️ This image may not be a proper brain MRI. Proceeding anyway...")

    # ---------------- PREDICTION ---------------- #

    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    svm_model, pca_model, vgg_model = load_models()

    with st.spinner("Analyzing MRI scan..."):
        features = vgg_model.predict(img_array)
        features = features.reshape(1, -1)

        features_pca = pca_model.transform(features)

        prediction = svm_model.predict(features_pca)
        probability = svm_model.predict_proba(features_pca)[0]

    st.subheader("📊 Prediction Results")
    predicted_class = classes[prediction[0]]
    st.success(f"Predicted Stage: {predicted_class}")

    st.subheader("📈 Confidence Scores")

    for label, prob in zip(classes, probability):
        st.write(f"{label}: {prob*100:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(classes, probability * 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("Model Confidence")
    ax.set_ylim(0, 100)

    for i, prob in enumerate(probability):
        ax.text(i, prob*100, f"{prob*100:.1f}%", ha='center')

    st.pyplot(fig)

st.info("System uses validation + ML prediction. Only grayscale MRI scans are strictly accepted.")
'''
st.title("APP IS WORKING")
st.write("If you see this, deployment is correct")
