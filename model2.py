from langchain_community.llms import Ollama
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

ollama = Ollama(base_url='http://localhost:11434', model='mistral')

pdf_path = 'data/anatomy_vol_1.pdf'
loader = PyMuPDFLoader(pdf_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())

question = 'Which bone is larger than the fibula?'
result = qachain({"query": question})
print(result)