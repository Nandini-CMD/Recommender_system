🧠 AI-Based Image Recommender System

A powerful and intuitive content-based image recommender system that uses deep learning to suggest visually similar images based on their features. This project combines ResNet50, FAISS, and Streamlit to deliver real-time recommendations with a clean and user-friendly interface.

🚀 Features
✅ Deep learning-powered feature extraction (ResNet50)

✅ Real-time image similarity recommendations

✅ Interactive web interface via Streamlit

✅ Batch-wise processing and FAISS-based indexing

✅ Supports .jpg, .jpeg, .png image uploads

🧩 How It Works
Feature Extraction:

Images are converted to 2048-dimensional feature vectors using a pretrained ResNet50 model.

Batch Processing:

app.py processes thousands of images in batches and stores their embeddings.

Indexing:

merge_batches.py merges batches into a single embedding file.

test.py uses FAISS for fast approximate nearest neighbor search.

Recommendation:

front.py powers a live Streamlit app where users upload an image and receive similar recommendations.

📂 Project Structure
bash
Copy
Edit
Recommender_system/
├── images/               # Original image dataset
├── uploads/              # Uploaded images via UI
├── batches/              # Processed image embedding batches
├── sample/               # Sample images for testing
├── embeddings.pkl        # Final feature embeddings
├── filenames.pkl         # Corresponding image filenames
├── app.py                # Batch-wise feature extractor
├── merge_batches.py      # Merges batches into final dataset
├── front.py              # Streamlit UI app
├── test.py               # FAISS-based test recommendations
├── requirements.txt      # Project dependencies
🛠️ Installation
⚙️ Prerequisites
Python ≥ 3.7

pip

(Optional) Virtualenv or conda

📦 Setup
bash
Copy
Edit
# Clone the repository
git clone https://github.com/Nandini-CMD/Recommender_system.git
cd Recommender_system

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
🧪 Usage
1. Embed Your Dataset (Only once)
bash
Copy
Edit
python app.py
python merge_batches.py
2. Launch Streamlit Web App
bash
Copy
Edit
streamlit run front.py
Then open http://localhost:8501 in your browser.

3. Test FAISS Search (Optional CLI)
bash
Copy
Edit
python test.py
📸 Example
Upload an image of a shoe, and get similar shoes recommended from your dataset!

(You can add a screenshot or gif here.)

🧠 Tech Stack
📷 ResNet50 – for feature extraction

⚡ FAISS – for fast similarity search

🖼️ Streamlit – for the frontend

🐍 Python – for the backend

🧑‍💻 Author
Nandini CMD
🌐 GitHub: @Nandini-CMD

