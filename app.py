import asyncio
import streamlit as st
import uuid
from mini_rag import rag_init, ask_question
APP_TITLE = "Personal RAG system"
USER_ID_COOKIE = "user_id"

def get_or_create_user_id() -> str:
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    user_id = str(uuid.uuid4())
    st.session_state[USER_ID_COOKIE] = user_id
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id

@st.cache_resource
def load_rag():
    return rag_init()

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE
    )   
    user_id = get_or_create_user_id()
    rag = load_rag()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['message'])

    user_input = st.chat_input()
    if user_input:
        st.session_state.messages.append(
            {
                "role": "human",
                "message" : user_input
            }
        )
        st.chat_message('human').write(user_input)
        response = ask_question(user_input, rag)
        st.session_state.messages.append(
            {
                "role": "ai",
                "message" : response['respond'].content
            }
        )
        st.chat_message('ai').write(response['respond'].content)
        
if __name__ == '__main__':
    main()
