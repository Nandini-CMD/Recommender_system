import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

features_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

if features_list.ndim == 3:
    features_list = features_list.reshape(features_list.shape[0], features_list.shape[2])
print(features_list.shape)

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalAveragePooling2D()
])

img = image.load_img('sample/1551.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)  
img_array = np.array(img_array, dtype=np.float32)  
preprocessed_img = preprocess_input(img_array)
result = model.predict(preprocessed_img)
normalized_result = result / norm(result)
normalized_result = normalized_result.reshape(1, -1)

neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
neighbours.fit(features_list)
distances, indices = neighbours.kneighbors(normalized_result)
print("Indices of nearest neighbors:", indices)


for file in indices[0]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (400, 400)))
    cv2.waitKey(0)