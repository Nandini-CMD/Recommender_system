import pickle
import os
import numpy as np

batch_dir = "batches"
all_features = []
all_filenames = []

batch_files = sorted(f for f in os.listdir(batch_dir) if f.startswith("embeddings_batch_"))

for file in batch_files:
    with open(os.path.join(batch_dir, file), "rb") as f:
        features, filenames = pickle.load(f)
        all_features.extend(features)
        all_filenames.extend(filenames)

features_np = np.array(all_features).squeeze().astype('float32')

with open("embeddings.pkl", "wb") as f:
    pickle.dump(features_np, f)

with open("filenames.pkl", "wb") as f:
    pickle.dump(all_filenames, f)

print(f" Merged {len(all_filenames)} images into embeddings.pkl and filenames.pkl")
