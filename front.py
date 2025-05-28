import streamlit as st
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
import pickle
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

# Load embeddings and filenames
features_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Convert to float32 and reshape if needed
features_list = np.array(features_list).astype('float32')
if features_list.ndim == 3:
    features_list = features_list.reshape(features_list.shape[0], features_list.shape[2])

# Load model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D()
])

# Streamlit UI
st.title('Image-Based Recommender System')

# Save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        os.makedirs("uploads", exist_ok=True)
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

# Feature extraction from uploaded image
def feature_extraction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = np.array(img_array, dtype=np.float32)
    preprocessed_img = preprocess_input(img_array)
    result = model.predict(preprocessed_img, verbose=0)
    normalized_result = result / norm(result)
    return normalized_result

# Recommend similar images
def recommend(features, features_list, filenames):
    neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbours.fit(features_list)
    distances, indices = neighbours.kneighbors(features)
    return [filenames[i] for i in indices[0]]

# Upload section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("File uploaded successfully!")
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', width=250)

        # Extract and recommend
        extracted_features = feature_extraction(os.path.join("uploads", uploaded_file.name))
        results = recommend(extracted_features, features_list, filenames)

        # Display results
        st.subheader("Recommended Images:")
        cols = st.columns(len(results))
        for i, file in enumerate(results):
            try:
                rec_img = Image.open(file)
                cols[i].image(rec_img, caption=f"Recommendation {i+1}", use_container_width=True)
            except:
                cols[i].write("Could not load image.")
    else:
        st.error("Failed to save file. Please try again.")
