import io
import os
import torch
import numpy as np
import clip
from PIL import Image
import pytesseract
from langchain.vectorstores import Chroma
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import streamlit as st
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize tesseract for OCR
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Function to extract images and text from a PDF using pdfplumber
pdf_path = "/content/anatomy_vol_2.pdf"
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
    def __init__(self):
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

def retrieve_images(query_embedding, top_k=5):
    query_embedding = np.array(query_embedding).flatten()
    image_scores = []

    for idx, emb in enumerate(image_embeddings):
        emb_array = np.array(emb['embedding']).flatten()
        score = compute_similarity(query_embedding, emb_array)
        image_scores.append((idx, score))

    image_scores.sort(key=lambda x: x[1], reverse=True)
    return image_scores[:top_k]

def run_image_query(query):
    query_embedding = clip_model.encode_text(clip.tokenize(query).to(device)).detach().cpu().numpy()
    top_images = retrieve_images(query_embedding)
    return top_images

def query_gemini(prompt):
    response = genai.generate_text(prompt=prompt)
    return response.result

# Function to handle text correction/enhancement
def handle_text_correction(prompt):
    if 'image' in prompt.lower() or 'picture' in prompt.lower():
        top_images = run_image_query(prompt)
        return top_images
    else:
        return query_gemini(prompt)

# Main function for Streamlit UI
def main():
    st.set_page_config(page_title="Anatomy 101", page_icon=":books:")

    with st.sidebar:
        st.subheader("Your Documents")

        preloaded_files = {
            "Anatomy_Basics.pdf": "https://drive.google.com/file/d/10wLiPf48SjQdYIM3_Z16d-sboRIauXGP/view?usp=drive_link",
            "Muscle_Structure.pdf": "https://drive.google.com/file/d/1fCUHEXMmLqy6uLVspj1T3OopWdExDn6j/view?usp=drive_link",
            "Skeletal_System.pdf": "https://drive.google.com/file/d/1N73KxQmCPMPiMwTqyZ4kfd5g8_WYlT4M/view?usp=drive_link"
        }

        st.write("Recommended Files:")
        for file_name, file_path in preloaded_files.items():
            file_link = f'<a href="{file_path}" target="_blank">{file_name}</a>'
            st.markdown(file_link, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload your PDFs here", type=["pdf"])

        if uploaded_file is not None:
            if 'vector_index' not in st.session_state:
                st.write(f"Uploaded file: {uploaded_file.name}")
                st.session_state['vector_index'] = process_pdf_async(uploaded_file)
                st.session_state['qa_chain'] = setup_qa_chain(st.session_state['vector_index'])
                st.success("File processed successfully!")

    st.title("Anatomy 101")
    st.header("Welcome to Anatomy 101")

    question = st.text_input("Ask a question!")

    if question:
        if 'qa_chain' in st.session_state:
            answer = answer_question(question, st.session_state['qa_chain'])
            st.write(f"Answer: {answer}")

            # Check for 'image' or 'picture' in the prompt
            if 'image' in question.lower() or 'picture' in question.lower():
                top_images = run_image_query(question)
                if top_images:
                    for idx, score in top_images:
                        image_data = image_embeddings[idx]['image_data']
                        st.image(image_data, caption=f"Image {idx} - Score: {score:.2f}")
                else:
                    st.warning("No images found for the query.")
            else:
                corrected_text = query_gemini(question)
                st.write(f"Corrected or enhanced text: {corrected_text}")

        else:
            st.warning("Please upload or select a document first.")
    
    # Flowchart generation section
    if st.button("Generate Flowchart"):
        med_query = answer
        flowchart_question = f"Can you help me with the process flow diagram for a {med_query}\
            Please use Graphviz DOT Language. Try to make it as detailed as possible with all the steps involved in the process.\
            Add colors to the different stages of the process to make it visually appealing."
        response = model1.generate_content(flowchart_question)

        formatted_dot_code = format_dot_code(response.text)

        png_bytes = get_png_bytes(formatted_dot_code)

        if png_bytes:
            st.image(png_bytes, caption="Generated Flowchart")
        else:
            st.error("Failed to generate the flowchart.")

    # Table generation section
    if st.button("Generate Table"):
        med_query = answer
        table_question = f"{med_query}.\n FOR THE GIVEN TEXT ABOVE, GENERATE A TABLE. THE TABLE SHOULD BE IN PRETTY TABLE PYTHON CODE. GIVE ONLY CODE"
        response = model1.generate_content(table_question)

        clean_code = response.text.strip('python').strip('')

        # Prepare a local variable dictionary to execute the code safely
        local_vars = {}

        # Execute the code and capture the table object in local_vars
        exec(clean_code, {"prettytable": prettytable}, local_vars)

        # Assuming the generated code creates a table object
        if 'table' in local_vars:
            table = local_vars['table']
            st.text(table)
        else:
            st.error("Failed to generate the table.")

if __name__ == '__main__':
    main()
