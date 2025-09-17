import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os 

class_names = ['airplane', 'automobile', 'bird', 
               'cat', 'deer', 'dog', 
               'frog', 'horse', 'ship', 'truck']

model_path='cifar.keras'

if not os.path.exists(model_path):
    st.error("Model not found")
    st.stop()

model=tf.keras.models.load_model(model_path)

st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption='Uploaded Image', use_container_width=True)

    img_resized = original_image.resize((32,32))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0) # (1,32,32,3)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f'Predicted Class: {predicted_class}')
