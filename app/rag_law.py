import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.config import EMBED_MODEL, QDRANT_URL, LAW_COLLECTION, LAW_PATH, CHUNK_SIZE, CHUNK_OVERLAP

embeddings = OllamaEmbeddings(model=EMBED_MODEL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def load_law_to_qdrant(pdf_path = LAW_PATH):
    law_text = extract_text_from_pdf(pdf_path)
    law_chunks = splitter.split_text(law_text)
    docs = [Document(page_content=chunk) for chunk in law_chunks]

    vectorstore = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=QDRANT_URL,
        collection_name=LAW_COLLECTION
    )
    return vectorstore

if __name__ == "__main__":
    print("Loading law PDF into Qdrant vector store...")
    vs = load_law_to_qdrant()
    print(f"Law loaded into collection '{LAW_COLLECTION}'.")