import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image

from numpy.linalg import norm
import numpy as np
import os
from tqdm import tqdm
import pickle

# Load ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalAveragePooling2D()])

# Prepare files
os.makedirs("batches", exist_ok=True)
image_dir = 'images'
filenames = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

# Process in batches
batch_size = 1000
for i in range(0, len(filenames), batch_size):
    batch_files = filenames[i:i+batch_size]
    features_list = []
    batch_filenames = []

    for file in tqdm(batch_files, desc=f"Batch {i}"):
        try:
            img = image.load_img(file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array = np.array(img_array, dtype=np.float32)
            preprocessed_img = preprocess_input(img_array)
            result = model.predict(preprocessed_img, verbose=0)
            normalized = result / norm(result)
            features_list.append(normalized.astype('float32'))
            batch_filenames.append(file)
        except Exception as e:
            print(f"Failed: {file} -> {e}")

    with open(f'batches/embeddings_batch_{i}.pkl', 'wb') as f:
        pickle.dump((features_list, batch_filenames), f)

print("All batches processed and saved.")
