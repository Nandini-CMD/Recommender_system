import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm 
import pickle


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])

def extract_features(img_path, model):
    """
    Extract features from an image using the pre-trained ResNet50 model.

    Args:
        image (numpy.ndarray): Input image of shape (224, 224, 3). 
           
    Returns:
        numpy.ndarray: Extracted features of shape (1, 2048).
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  
    img_array = np.array(img_array, dtype=np.float32)  
    preprocessed_img = preprocess_input(img_array)
    result = model.predict(preprocessed_img)
    normalized_result = result / norm(result)
    return normalized_result

filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

features_list = []
for file in tqdm(filenames):
    features = extract_features(file, model)
    features_list.append(features)

pickle.dump(features_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
