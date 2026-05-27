from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any
from chroma_database import load_database
from langchain_chroma import Chroma


RAG_DIR = './data'
retriever: VectorStoreRetriever | None= None
response_model: BaseChatModel | None = None

class RAGState(MessagesState):
    docs: list[Document]
    question: str
    respond: AIMessage | None
        
def ask_question(question: str, rag_agent: CompiledStateGraph) -> RAGState:
    result = rag_agent.invoke({
        'docs': [],
        'question': question,
        "respond": None
    })
    return result

def rag_init(data_path: str = './data/note.txt') -> tuple[CompiledStateGraph, Chroma]:
    global response_model, retriever
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
    def retrieve_data(state: RAGState) -> dict[str, list[Document]]:
        """Search data"""
        docs = retriever.invoke(state['question'])
        return {'docs': docs}
    
    chroma = load_database()
    retriever = chroma.as_retriever(search_kwargs={"k": 3})
    response_model = init_chat_model("gpt-5-nano", temperature=0)
    agent = StateGraph(RAGState)
    agent.add_node("retrieve", retrieve_data)
    agent.add_node("respond", generate_respond)
    agent.set_entry_point("retrieve")
    agent.add_edge("retrieve", "respond")
    agent.add_edge("respond", END)
    rag_agent = agent.compile()
    return rag_agent, chroma