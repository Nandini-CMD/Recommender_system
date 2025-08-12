# 🖼️LOOKLENS Image Based Recommender System

This project implements a **content-based image recommendation system** powered by deep learning. By leveraging **ResNet50** for visual feature extraction and **FAISS** for fast similarity search, this system suggests visually similar images based on the one uploaded by the user. The app is presented through an interactive **Streamlit** interface.

## 🌟 Key Features

- 🔍 Content-based image recommendations using CNN feature vectors
- ⚙️ ResNet50 backbone (pretrained on ImageNet) for deep feature extraction
- ⚡ Fast similarity search using **FAISS** and **Nearest Neighbors**
- 🧵 Batch-wise embedding creation and merging for large datasets
- 💻 Streamlit web interface for easy image upload and result display

## 🧠 Model Architecture and Purpose

### 📷 ResNet50 (Feature Extraction)

- **Model**: ResNet50, a deep Convolutional Neural Network (CNN) pretrained on the **ImageNet** dataset.
- **Usage**: Extracts a **2048-dimensional feature vector** from each image by removing the top classification layers (`include_top=False`) and applying a **GlobalAveragePooling2D** layer.
- **Why ResNet50?**
  - Deep residual connections help retain semantic information
  - Robust and efficient for capturing fine visual features
  - Pretrained on ImageNet, so it's generalized for many image types

```python
model = Sequential([
    ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    GlobalAveragePooling2D()
])
```

### 🧭 FAISS / NearestNeighbors (Similarity Search)

- **Library**: [FAISS](https://github.com/facebookresearch/faiss) and `sklearn.neighbors.NearestNeighbors`
- **Usage**: Finds the **top K nearest neighbors** in the embedding space based on **Euclidean distance**.
- **Why FAISS?**
  - Optimized for large-scale similarity searches
  - Great performance on high-dimensional data like image embeddings

## 🗂️ Project Structure

```
Recommender_system/
├── images/               # Input image dataset
├── uploads/              # Uploaded images via the web interface
├── batches/              # Intermediate embedding batches
├── sample/               # Sample test images
├── embeddings.pkl        # Merged embedding vectors
├── filenames.pkl         # Image paths corresponding to embeddings
├── app.py                # Extracts embeddings in batch using ResNet50
├── merge_batches.py      # Merges batch outputs into final embedding files
├── front.py              # Streamlit-based front-end
├── test.py               # FAISS-based CLI similarity test
├── requirements.txt      # List of dependencies
```

## 🛠️ Installation & Setup

### 🔧 Prerequisites

- Python ≥ 3.7
- pip
- Virtual environment (optional but recommended)

### 📦 Steps

1. **Clone the repository**

```bash
git clone https://github.com/Nandini-CMD/Recommender_system.git
cd Recommender_system
```

2. **(Optional) Set up a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🚀 Usage Instructions

### Step 1: Create Embeddings from Images

```bash
python app.py
```

### Step 2: Merge Embedding Batches

```bash
python merge_batches.py
```

### Step 3: Launch the Streamlit Web App

```bash
streamlit run front.py
```

Navigate to `http://localhost:8501` in your browser.

### (Optional) Step 4: Test FAISS-Based Search from Command Line

```bash
python test.py
```

## 🧪 Example Output

> Uploading a photo of a shoe will return images of similar shoes from the dataset, based on visual features—not labels or tags.

## 🧰 Tech Stack

| Component      | Description                          |
|----------------|--------------------------------------|
| 🐍 Python      | Primary programming language         |
| 🧠 ResNet50    | CNN used for image feature extraction |
| ⚡ FAISS       | Fast approximate similarity search    |
| 🧪 Scikit-learn | Brute-force Nearest Neighbors (Web UI) |
| 🖼️ Streamlit  | Web interface for image upload and display |

