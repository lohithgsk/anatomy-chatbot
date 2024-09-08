import io
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import webbrowser
import concurrent.futures
import prettytable
import torch
import streamlit as st
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Detect if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to format DOT code
def format_dot_code(dot_code: str) -> str:
    formatted_code = dot_code.strip("```dot").strip()
    lines = formatted_code.split("\n")
    for i, line in enumerate(lines):
        if "rankdir" in line:
            lines[i] = "    rankdir=TB;"
    return "\n".join(lines)

# Function to get PNG from the API and return it as bytes
def get_png_bytes(dot_code: str) -> bytes:
    quickchart_url = "https://quickchart.io/graphviz"
    post_data = {"graph": dot_code, "format": "png"}

    try:
        response = requests.post(quickchart_url, json=post_data, verify=False)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "image/png" in content_type:
            return response.content
        else:
            print("Unexpected response content type:", content_type)
            return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

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
        else:
            st.warning("Please upload or select a document first.")
    
    genai.configure(api_key=API_KEY)
    model1 = genai.GenerativeModel("gemini-1.5-flash-latest")

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

        clean_code = response.text.strip('```python').strip('```')

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
