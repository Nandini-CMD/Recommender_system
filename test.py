import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from numpy.linalg import norm
import faiss
import cv2

# Load data
features_list = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))
features_list = np.array(features_list).astype('float32')

# Load model
model = tf.keras.Sequential([
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    GlobalAveragePooling2D()
])

# Feature extraction
img = image.load_img('sample/1551.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)
img_array = np.array(img_array, dtype=np.float32)
preprocessed = preprocess_input(img_array)
result = model.predict(preprocessed, verbose=0)
normalized = result / norm(result)
normalized = normalized.astype('float32')

# FAISS
index = faiss.IndexFlatL2(features_list.shape[1])
index.add(features_list)
distances, indices = index.search(normalized, k=5)

# Display
for idx in indices[0]:
    img = cv2.imread(filenames[idx])
    cv2.imshow("Recommendation", cv2.resize(img, (400, 400)))
    cv2.waitKey(0)
