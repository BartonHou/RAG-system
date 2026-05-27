from fastapi import FastAPI, APIRouter
from mini_rag import ask_question, rag_init, chunk_file, add_files
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
rag = rag_init()

class AskQuestion(BaseModel):
    question: str

    
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
        with open(path, "wb") as f:
            f.write(file.file.read())
        loader = PDFPlumberLoader(path)
        documents = loader.load()
        chunks = chunk_file(documents)
        add_files(chunks)
        return {
            "status": "ok",
            "file_name": filename, 
            "saved_path": path,
            "pages": len(documents),
            "chunks": len(chunks)
        }
    except Exception as e:
        print(f"Error occur while loading pdf file {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/health")
def health_check():
    health_status = {"status": "ok"}
    return health_status

app.include_router(router)