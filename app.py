import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.load_model('vegetable_model.h5') # অথবা tf.keras.models.load_model
    return model

model = tf.keras.models.load_model('vegetable_model.h5')
class_names = ['Bean', 'Bitter melon', 'Brinjal', 'Cucumber', 'Garlic', 'Green Chili', 'Ladies finger', 'Onion', 'Pointed gourd', 'Potato', 'Radish', 'Tomato']

st.title("Vegetable Classification App")
st.write("সবজির ছবি আপলোড করুন:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    st.success(f"Predicted Vegetable: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")