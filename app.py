import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import concurrent.futures
import torch

# Load API key from environment
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Detect if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and extract text from PDF
def extract_pdf_text(uploaded_file):
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            pages.append(text)
    return "\n\n".join(pages)

# Function to split the text into chunks for embeddings
def split_text(text, chunk_size=5000, chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# Function to create the vector store (only needed once, can be stored and reused)
def create_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})  # Top 3 results for faster search
    return vector_index

# Asynchronous function to process the PDF and create the vector store
def process_pdf_async(uploaded_file):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(process_pdf, uploaded_file)
        return future.result()

# Synchronous version of processing PDF
def process_pdf(uploaded_file):
    context = extract_pdf_text(uploaded_file)
    texts = split_text(context)
    vector_index = create_vector_store(texts)
    return vector_index

# Function to set up the QA chain
def setup_qa_chain(vector_index):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY, temperature=0.2, convert_system_message_to_human=True)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True
    )
    return qa_chain

# Function to handle question answering
def answer_question(question, qa_chain):
    result = qa_chain({"query": question})
    return result["result"]

# Main function for Streamlit UI
def main():
    # Set page configuration
    st.set_page_config(page_title="Anatomy 101", page_icon=":books:")

    # Sidebar for file uploads and preloaded files
    with st.sidebar:
        st.subheader("Your Documents")

        # Preloaded files (you can replace these with actual file paths)
        preloaded_files = {
            "Anatomy_Basics.pdf": "https://drive.google.com/file/d/10wLiPf48SjQdYIM3_Z16d-sboRIauXGP/view?usp=drive_link",
            "Muscle_Structure.pdf": "https://drive.google.com/file/d/1fCUHEXMmLqy6uLVspj1T3OopWdExDn6j/view?usp=drive_link",
            "Skeletal_System.pdf": "https://drive.google.com/file/d/1N73KxQmCPMPiMwTqyZ4kfd5g8_WYlT4M/view?usp=drive_link"
        }

        # Display preloaded files as links
        st.write("Recommended Files:")
        for file_name, file_path in preloaded_files.items():
            file_link = f'<a href="{file_path}" target="_blank">{file_name}</a>'
            st.markdown(file_link, unsafe_allow_html=True)

        # File uploader for new PDFs
        uploaded_file = st.file_uploader("Upload your PDFs here", type=["pdf"])

        # Asynchronous PDF processing and vector index creation
        if uploaded_file is not None:
            if 'vector_index' not in st.session_state:
                st.write(f"Uploaded file: {uploaded_file.name}")
                st.session_state['vector_index'] = process_pdf_async(uploaded_file)
                st.session_state['qa_chain'] = setup_qa_chain(st.session_state['vector_index'])
                st.success("File processed successfully!")

    # Main content area
    st.title("Anatomy 101")
    st.header("Welcome to Anatomy 101")

    # Ask a question
    question = st.text_input("Ask a question!")

    # Show answer when a question is asked
    if question:
        if 'qa_chain' in st.session_state:
            # Answer question using the pre-built QA chain
            answer = answer_question(question, st.session_state['qa_chain'])
            st.write(f"Answer: {answer}")
        else:
            st.warning("Please upload or select a document first.")

if __name__ == '__main__':
    main()
