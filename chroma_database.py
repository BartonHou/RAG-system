import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

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

def add_files(chroma: Chroma, documents: list[Document]) -> None:
    chroma.add_documents(documents)