from fastapi import FastAPI, APIRouter
from mini_rag import ask_question, rag_init
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

app = FastAPI()
router = APIRouter()
rag = rag_init()

class AskQuestion(BaseModel):
    question: str

@router.post("/invoke")
def invoke(quest: AskQuestion):
    question = quest.question
    response = ask_question(question, rag)
    answer = response['respond'].content
    return {'answer': answer}

@app.get("/health")
def health_check():
    health_status = {"status": "ok"}
    return health_status

app.include_router(router)