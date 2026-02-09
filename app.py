import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 
import os
import io

# --- Configuration (MUST MATCH train_model.py) ---
MODEL_PATH = 'deepfake_cae_model.h5'
IMG_SIZE = 128
# >>> ANOMALY THRESHOLD <<<
# This value separates Real (low error) from Fake (high error).
RECONSTRUCTION_ERROR_THRESHOLD = 0.0005 

# --- Model Loading (Cached for fast performance) ---
@st.cache_resource
def load_model():
    """
    Loads the trained Keras model, cached for efficiency.
    FIXED: Uses custom_objects to prevent the 'deserialize mse' error.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please run train_model.py first.")
        st.stop()
    try:
        # CRITICAL FIX: Explicitly pass custom_objects to handle the MSE loss correctly
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Face Detector (Cached) ---
@st.cache_resource
def load_face_detector():
    """Loads the OpenCV Haar Cascade classifier for face detection."""
    # Ensure the path is correct for the default OpenCV cascade file
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Error: Could not load OpenCV Haar Cascade classifier.")
        st.stop()
    return face_cascade

# --- Prediction Function ---
def predict_deepfake(image, model, face_cascade):
    """Detects a face, runs the autoencoder, and calculates MSE for anomaly detection."""
    # Convert PIL Image to OpenCV format (BGR)
    img_cv = np.array(image.convert('RGB'))
    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    if len(faces) == 0:
        return None, "No Face Detected", "Please upload an image containing a clear face.", None

    # Process the largest face found
    (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0] 
    face_crop = img_cv[y:y+h, x:x+w]
    
    # Resize and normalize for the model (128x128 and [0, 1])
    face_pil = Image.fromarray(face_crop)
    face_resized = face_pil.resize((IMG_SIZE, IMG_SIZE))
    face_array = np.array(face_resized) / 255.0
    input_data = np.expand_dims(face_array, axis=0) # Add batch dimension

    # Autoencoder Reconstruction
    reconstructed_face = model.predict(input_data, verbose=0)[0]
    
    # Calculate Reconstruction Error (Mean Squared Error)
    mse = np.mean(np.square(face_array - reconstructed_face))

    # Anomaly Detection Logic
    if mse > RECONSTRUCTION_ERROR_THRESHOLD:
        result_text = "⚠️ DEEPFAKE DETECTED"
        st.error(f"{result_text}")
    else:
        result_text = "✅ REAL IMAGE"
        st.success(f"{result_text}")
    
    confidence_level = f"Reconstruction Error (MSE): **{mse:.6f}**"
    
    # Convert reconstructed array back to PIL Image for display
    reconstructed_face_display = Image.fromarray((reconstructed_face * 255).astype('uint8'))
    
    return face_resized, result_text, confidence_level, reconstructed_face_display


# --- Streamlit UI ---
st.set_page_config(page_title="DeepFake Detector (Autoencoder)", layout="wide")

st.title("🤖 DeepFake Anomaly Detector (CAE)")
st.markdown(f"""
    This detector uses a Convolutional Autoencoder trained **only on Real faces**. 
    If the image is an anomaly (fake), the reconstruction error (MSE) will be high.
""")
st.markdown(f"**Current Anomaly Threshold:** `{RECONSTRUCTION_ERROR_THRESHOLD:.6f}`")
st.markdown("---")

col1, col2 = st.columns([1, 1])

# Column 1: File Uploader and Controls
with col1:
    st.subheader("Upload Image for Analysis")
    uploaded_file = st.file_uploader("Choose an image (JPEG, PNG)", type=["jpg", "png", "jpeg"])
    
    detector_model = load_model()
    face_detector = load_face_detector()

    if uploaded_file is not None and detector_model is not None:
        image = Image.open(uploaded_file)
        
        # NOTE: The button only appears after a file is uploaded and the model is loaded.
        if st.button("Run DeepFake Detection", use_container_width=True):
            with st.spinner('Analyzing Image and Detecting Face...'):
                face_original, result, confidence, face_reconstructed = predict_deepfake(
                    image, detector_model, face_detector
                )
            
            # Save results in session state for column 2 display
            st.session_state['result'] = result
            st.session_state['confidence'] = confidence
            st.session_state['face_original'] = face_original
            st.session_state['face_reconstructed'] = face_reconstructed
            st.session_state['uploaded_file_name'] = uploaded_file.name
        
        st.markdown("---")
        st.image(image, caption='Uploaded Image', use_column_width=True)


# Column 2: Results Display
with col2:
    st.subheader("Detection Results")
    if 'face_original' in st.session_state and st.session_state['face_original'] is not None:
        st.markdown(f"#### {st.session_state['result']}")
        st.markdown(st.session_state['confidence'])
        st.caption(f"If the error is **above** {RECONSTRUCTION_ERROR_THRESHOLD:.6f}, it's flagged as an anomaly/deepfake.")
        
        st.markdown("---")
        
        # Display the cropped original and the reconstructed version
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.image(st.session_state['face_original'], caption='Input: Detected Face', use_container_width=True)
        with result_col2:
            st.image(st.session_state['face_reconstructed'], caption='Output: Reconstructed Face', use_container_width=True)
        
        # Explain the principle
        st.info("The difference between the Input and the Reconstructed Output measures how 'normal' the face is according to the model's training. A large difference (high MSE) indicates the presence of forgery artifacts the model hasn't seen before.")
    
    elif 'result' in st.session_state and st.session_state['face_original'] is None:
        st.warning(st.session_state['confidence']) 
    else:
        st.info("Upload an image and click 'Run DeepFake Detection' to see results.")