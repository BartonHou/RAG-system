import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import hashlib
import uuid
load_dotenv()

def load_database(db_name: str = "./chroma_db") -> Chroma:
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    chroma = Chroma(
        embedding_function=embeddings, 
        persist_directory=db_name
    )
    return chroma

def chunk_file(documents: list[Document], chunkSize: int, overlap: int = None):

    if overlap is None:
        overlap = chunkSize // 2
    if chunkSize <= overlap:
        raise ValueError("Error: Overlap should not be greater than or equal to chunk size")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunkSize, chunk_overlap=overlap
    )
    return text_splitter.split_documents(documents)

def add_files(filename: str, path: str, chroma: Chroma, documents: list[Document], file_hash: str) -> str:
    doc_id = str(uuid.uuid4())
    ids = []
    for chunk_id, document in enumerate(documents):
        document.metadata.update({
            "file_id": doc_id,
            "file_name": filename,
            "chunk": chunk_id,
            "path": path,
            "file_hash": file_hash
        })
        ids.append(f"{doc_id}:{chunk_id}")
    chroma.add_documents(documents, ids=ids)
    return doc_id

def compute_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def find_file_by_hash(chroma: Chroma, file_hash: str) -> dict:
    existing = chroma.get(
        where={"file_hash": file_hash},
        include=["metadatas"]
    )

    if not existing["ids"]:
        return None

    metadata = existing["metadatas"][0]

    return {
        "file_id": metadata.get("file_id"),
        "file_name": metadata.get("file_name"),
        "path": metadata.get("path"),
        "file_hash": metadata.get("file_hash"),
        "n_existing_chunks": len(existing["ids"]),
    } 