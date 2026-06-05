from fastapi import FastAPI, APIRouter
from mini_rag import ask_question, rag_init
from chroma_database import (
    chunk_file, 
    add_files, 
    compute_file_hash,
    find_file_by_hash
)
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from fastapi import UploadFile
from fastapi import File
from langchain_community.document_loaders import PDFPlumberLoader

import os 

RAG_DIR = './data'
app = FastAPI()
router = APIRouter()

print("INITIATING RAG")
rag, chroma = rag_init()

class AskQuestion(BaseModel):
    question: str

class SearchRequest(BaseModel):
    question: str
    k: int = 3
    file_id: str | None = None

@router.post("/invoke")
def invoke(quest: AskQuestion):
    question = quest.question
    response = ask_question(question, rag)
    answer = response['respond'].content
    return {'answer': answer}

@router.post("/upload")
def get_file(file: UploadFile = File(...)):
    try:
        filename = file.filename
        path = os.path.join(RAG_DIR, filename)
        os.makedirs(RAG_DIR, exist_ok=True)

        content = file.file.read()
        file_hash = compute_file_hash(content)

        existing_file = find_file_by_hash(chroma, file_hash)
        if existing_file is not None:
            return {
                "status": "exists",
                "message": "File already exists in vector database",
                "file_id": existing_file["file_id"],
                "file_name": existing_file["file_name"],
                "path": existing_file["path"],
                "file_hash": existing_file["file_hash"],
                "n_chunk": existing_file["n_existing_chunks"],
            }
        
        with open(path, "wb") as f:
            f.write(content)
        loader = PDFPlumberLoader(path)
        documents = loader.load()
        chunks = chunk_file(documents, 500)
        file_id = add_files(filename, path, chroma, chunks, file_hash)
        return {
            "status": "ok",
            "file_id": file_id,
            "file_name": filename, 
            "file_hash": file_hash,
            "path": path,
            "n_doc": len(documents),
            "n_chunk": len(chunks)
        }
    except Exception as e:
        print(f"Error occur while loading pdf file {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/files/{file_id}")
def query_by_file_id(file_id: str):
    return chroma.get(
        where={
            "file_id": file_id
        }
    ) 
@app.get("/id/{id}")
def query_by_chunk_id(id: str):
    return chroma.get(ids=[id]) 

@app.get("/chunks")
def get_all_chunks():
    data = chroma.get(include=["metadatas", "documents"])
    chunks = []
    for chroma_id, metadata, document in zip(data["ids"], data["metadatas"], data["documents"]):
        if metadata is None:
            continue
        file_id = metadata.get("file_id")
        file_name = metadata.get("file_name")
        path = metadata.get("path")
        chunk_id = metadata.get("chunk")
        chunks.append({
            "id": chroma_id,
            "file_id": file_id,
            "file_name": file_name,
            "chunk_id": chunk_id,
            "path": path,
            "document": document
        })
    return chunks

@app.get("/files")
def list_files():
    data = chroma.get(include=["metadatas"])

    files = {}
    for chroma_id, metadata in zip(data["ids"], data["metadatas"]):
        if metadata is None:
            continue
        file_id = metadata.get("file_id")
        file_name = metadata.get("file_name")
        path = metadata.get("path")
        if file_id not in files:
            files[file_id] = {
                "file_id": file_id,
                "file_name": file_name,
                "path": path,
                "n_chunks": 0,
                "ids": []
            }
        files[file_id]["n_chunks"] += 1
        files[file_id]['ids'].append(chroma_id)

    return {
        "total_files": len(files),
        "files": list(files.values())
    }

@app.delete("/files/{file_id}")
def delete_file(file_id: str):
    existing = chroma.get(
        where={"file_id": file_id},
        include=["metadatas"]
    )

    ids = existing["ids"]

    if not ids:
        return {
            "status": "not_found",
            "file_id": file_id,
            "deleted_chunks": 0
        }

    chroma.delete(ids=ids)

    return {
        "status": "ok",
        "file_id": file_id,
        "deleted_chunks": len(ids)
    }


@app.post("/debug/search")
def debug_search(req: SearchRequest):
    if req.file_id:
        docs = chroma.similarity_search(
            req.question,
            k=req.k,
            filter={"file_id": req.file_id}
        )
    else:
        docs = chroma.similarity_search(
            req.question,
            k=req.k
        )

    return {
        "query": req.question,
        "file_id": req.file_id,
        "retrieved": len(docs),
        "total_in_db": chroma._collection.count(),
        "docs": [
            {
                "content": doc.page_content[:1000],
                "metadata": doc.metadata
            }
            for doc in docs
        ]
    }

@app.get("/health")
def health_check():
    health_status = {"status": "ok"}
    return health_status

app.include_router(router)