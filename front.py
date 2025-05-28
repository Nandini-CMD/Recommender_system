import streamlit as st
import os
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

features_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

if features_list.ndim == 3:
    features_list = features_list.reshape(features_list.shape[0], features_list.shape[2])


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])

st.title('Recommender System Project')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  
    img_array = np.array(img_array, dtype=np.float32)  
    preprocessed_img = preprocess_input(img_array)
    result = model.predict(preprocessed_img)
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, features_list, filenames):
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbours.fit(features_list)
    distances, indices = neighbours.kneighbors(features)
    return [filenames[i] for i in indices[0]]

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("File uploaded successfully!")
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', width=200)

        # feature extraction
        extracted_features = feature_extraction(os.path.join("uploads", uploaded_file.name))
        
        # recommendation
        indicies = recommend(extracted_features, features_list, filenames)
        # Display recommended images
        st.subheader("Recommended Images:")
        cols = st.columns(len(indicies))  # Create one column per image

        for i, file in enumerate(indicies):
            img = Image.open(file)
            cols[i].image(img, caption=f"Recommended {i+1}", use_container_width=True)

    else:
        st.error("Failed to upload file. Please try again.")

    