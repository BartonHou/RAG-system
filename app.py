import streamlit as st
import uuid
from langgraph.graph.state import CompiledStateGraph
import httpx
import os
APP_TITLE = "Personal RAG system"
USER_ID_COOKIE = "user_id"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


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


class ClientError(Exception):
    pass

def invoke(question: str, url: str= API_URL):
    try:
        response = httpx.post(url=f'{url}/invoke', json={"question": question}, timeout=300)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise ClientError(f"request fail {e}")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE
    )   

    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        st.rerun()

    user_id = get_or_create_user_id()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.chat_message('ai').write('Hello, my lord. I am your loyal personal slave. Master, Ask me anything')
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
        with st.spinner("Fabricating..."):
            answer = invoke(user_input)['answer']
        st.session_state.messages.append(
            {
                "role": "ai",
                "message" : answer
            }
        )
        st.chat_message('ai').write(answer)
        
if __name__ == '__main__':
    main()
