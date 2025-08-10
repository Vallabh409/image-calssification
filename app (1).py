
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import gdown
import os

# Step 1: Download model from Google Drive if not already present
file_id = '169YCYaS1pEiRyo2bRXHK5D0pYBVHwZ7j'
output_file = 'best_model.h5'

if not os.path.exists(output_file):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)

# Step 2: Load the model
model = tf.keras.models.load_model(output_file)

# Step 3: Define class names
class_names = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food trout',
    'fish sea_food house_mackarel', 'fish sea_food red_mullet',
    'fish sea_food red_sea_bream', 'fish sea_food sea_bass',
    'fish sea_food shrimp', 'fish sea_food striped_red_mullet'
]

# Step 4: Streamlit UI
st.title("🐟 Fish Species Classifier")
st.write("Upload a fish image and the model will predict its category.")

uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption='🖼️ Uploaded Image', use_column_width=True)

        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index]

        # Display result
        st.markdown(f"### 🐠 Predicted Class: **{predicted_class}**")
        st.markdown(f"### 🔍 Confidence: **{confidence:.2f}**")

        # Display all class scores
        st.markdown("### 📊 Class Probabilities:")
        for i, score in enumerate(predictions):
            st.write(f"{class_names[i]}: {score:.2f}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
else:
    st.info("👉 Please upload a fish image to classify.")
