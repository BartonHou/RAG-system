from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os 

load_dotenv()


class RAGState(MessagesState):
    docs: list[Document]
    question: str
    respond: AIMessage | None

def read_file(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def chunk_the_file(file: str, chunkSize: int, overlap: int = None) -> list[str]:
    chunk = []
    start = 0
    if overlap is None:
        overlap = chunkSize // 2

    if chunkSize <= overlap:
        raise ValueError("Error: Overlap should not be greater than or equal to chunk size")

    while start < len(file):
        chunk.append(file[start: start + chunkSize])
        start+=(chunkSize - overlap)
    return chunk

def build_retriever(chunks: list[str], model: str) -> VectorStoreRetriever:
    embedding = OpenAIEmbeddings(model=model)
    docs = []
    for chunk in chunks:
        docs.append(Document(page_content=chunk))
    vectorstate = InMemoryVectorStore.from_documents(
        documents=docs, embedding=embedding
    )
    return vectorstate.as_retriever(search_kwargs={"k": 3})

def rag_init(data_path: str = './data/note.txt') -> CompiledStateGraph:
    def retrieve_data(state: RAGState) -> dict[str, list[Document]]:
        """Search data"""
        docs = retriever.invoke(state['question'])
        return {'docs': docs}
    def generate_respond(state: RAGState) -> dict[str, AIMessage]:
        context = "\n\n".join(doc.page_content for doc in state['docs'])

        prompt = f'''
        You are my loyal personal slave. Before answering the question, you should greet me with 'My lord'. 
        This is the context of the problem you are given
        {context},
        And this this the question you now been asked, 
        {state['question']},
        If there is no answer existed in context, You must answer: forgive my stupidity, I do not know.
        Do not use your own knowledge. 
        '''
        respond = response_model.invoke(prompt)
        return {'respond': respond}

    entirefile = read_file(data_path)
    chunks = chunk_the_file(entirefile, 500)
    retriever = build_retriever(chunks, "text-embedding-3-small")
    response_model = init_chat_model("gpt-5-nano", temperature=0)
    agent = StateGraph(RAGState)
    agent.add_node("retrieve", retrieve_data)
    agent.add_node("respond", generate_respond)
    agent.set_entry_point("retrieve")
    agent.add_edge("retrieve", "respond")
    agent.add_edge("respond", END)
    rag_agent = agent.compile()
    return rag_agent

def ask_question(question: str, rag_agent: CompiledStateGraph) -> RAGState:
    result = rag_agent.invoke({
        'docs': [],
        'question': question,
        "respond": None
    })
    return result