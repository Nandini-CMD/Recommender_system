import streamlit as st
import os

st.title('Recommender System Project')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="visible")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("File uploaded successfully!")
    else:
        st.error("Failed to upload file. Please try again.")
    