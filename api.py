import pdfplumber
import io
import torch
import numpy as np
import clip
from PIL import Image
import pytesseract
from langchain.vectorstores import Chroma
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os
import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

# Configure the Gemini API

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize tesseract for OCR
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Function to extract images and text from a PDF using pdfplumber

pdf_path = "book/anatomy_vol_2.pdf"
images_data = []

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        for img in page.images:
            img_bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
            img_cropped = page.within_bbox(img_bbox).to_image()

            image_bytes = io.BytesIO()
            img_cropped.save(image_bytes, format="PNG")

            # Extract text from image using OCR (if applicable)
            image_pil = Image.open(io.BytesIO(image_bytes.getvalue()))
            recognized_text = pytesseract.image_to_string(image_pil)

            images_data.append({
                "page_number": i + 1,
                "image_data": image_bytes.getvalue(),
                "bbox": img_bbox,
                "recognized_text": recognized_text
            })

# Process images to obtain CLIP embeddings
image_embeddings = []
for img_data in images_data:
    image = Image.open(io.BytesIO(img_data['image_data']))
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_embedding = clip_model.encode_image(image_input).cpu().numpy()

    image_embeddings.append({
        "page_number": img_data["page_number"],
        "embedding": img_embedding,
        "image_data": img_data["image_data"],
        "recognized_text": img_data["recognized_text"]
    })

class InMemoryImageStore:
    def _init_(self):
        self.image_embeddings = []
        self.image_metadata = []

    def add_image(self, embedding, metadata):
        self.image_embeddings.append(embedding)
        self.image_metadata.append(metadata)

    def query(self, query_embedding, top_k=5):
        def compute_similarity(a, b):
            a_flat = a.flatten()
            b_flat = b.flatten()
            return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

        image_scores = [(idx, compute_similarity(query_embedding, emb)) for idx, emb in enumerate(self.image_embeddings)]
        image_scores.sort(key=lambda x: x[1], reverse=True)
        return image_scores[:top_k]

# Create the image store and add images with their embeddings
image_store = InMemoryImageStore()
for img_data in image_embeddings:
    image_store.add_image(img_data["embedding"], img_data["image_data"])

from sklearn.metrics.pairwise import cosine_similarity
def retrieve_images(query_embedding, top_k=5):
    query_embedding = np.array(query_embedding).flatten()
    image_scores = []

    for idx, emb in enumerate(image_embeddings):
        emb_array = np.array(emb['embedding']).flatten()
        score = compute_similarity(query_embedding, emb_array)
        image_scores.append((idx, score))

    image_scores.sort(key=lambda x: x[1], reverse=True)
    return image_scores[:top_k]

# Display the retrieved images
def display_images(image_results):
    for idx, score in image_results:
        image_data = image_embeddings[idx]['image_data']
        try:
            image = Image.open(io.BytesIO(image_data))
            plt.figure()
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Image {idx} - Score: {score:.2f}")
            plt.show()
        except Exception as e:
            print(f"Error displaying the image: {e}")

# Function to query the Gemini API for text-based corrections or enhancements
def query_gemini(prompt):
    response = genai.generate_text(prompt=prompt)
    return response.result

def match_dimensions(a, b):
    """Adjust the dimensions of vectors a and b to match."""
    if a.shape[0] == b.shape[0]:
        return a, b

    if a.shape[0] < b.shape[0]:
        b = np.interp(np.linspace(0, 1, a.shape[0]), np.linspace(0, 1, b.shape[0]), b)
    else:
        a = np.interp(np.linspace(0, 1, b.shape[0]), np.linspace(0, 1, a.shape[0]), a)

    return a, b

def compute_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a, b = match_dimensions(np.array(a).flatten(), np.array(b).flatten())
    return cosine_similarity([a], [b])[0][0]

def query_gemini(prompt):
    response = genai.generate_text(prompt=prompt)
    return response.result

def run_image_query(query):
    query_embedding = clip_model.encode_text(clip.tokenize(query).to(device)).detach().cpu().numpy()
    top_images = retrieve_images(query_embedding)
    return top_images

# Example usage
question = "lungs"
top_images = run_image_query(question)

# Display retrieved images
display_images(top_images)

# Query Gemini API for image corrections/enhancements
for idx, _ in top_images:
    recognized_text = image_embeddings[idx]["recognized_text"]
    if recognized_text.strip():
        prompt = f"Correct the following image text: {recognized_text}"
        corrected_text = query_gemini(prompt)
        print(f"Corrected text for image {idx}: {corrected_text}")