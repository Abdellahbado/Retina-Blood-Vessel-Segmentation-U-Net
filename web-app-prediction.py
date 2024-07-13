import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('best_model.keras')

def preprocess_image(image):
    # Resize the image to match the input size of your model
    image = image.resize((512, 512))
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_mask(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(preprocessed_image)
    # Convert prediction to binary mask
    mask = (prediction > 0.5).astype(np.uint8)
    return mask[0, :, :, 0]

st.title('Image Segmentation App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    if st.button('Segment Image'):
        mask = predict_mask(image)
        
        # Display the segmentation mask
        st.image(mask * 255, caption='Segmentation Mask', use_column_width=True)

        # Overlay the mask on the original image
        overlay = np.zeros((512, 512, 3), dtype=np.uint8)
        overlay[:, :, 1] = mask * 255  # Green channel
        overlay_image = Image.fromarray(overlay).convert('RGBA')
        original_image = image.resize((512, 512)).convert('RGBA')
        
        blended = Image.blend(original_image, overlay_image, alpha=0.5)
        st.image(blended, caption='Segmentation Overlay', use_column_width=True)