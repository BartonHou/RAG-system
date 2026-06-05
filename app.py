import streamlit as st
import uuid
from langgraph.graph.state import CompiledStateGraph
import httpx
import os
from typing import Any




APP_TITLE = "Personal RAG system"
USER_ID_COOKIE = "user_id"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
WELCOME = 'Hello, my lord. I am your loyal personal slave. Master, Ask me anything'

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

def invoke(question: str,  timeout: float, url: str= API_URL):
    try:
        request_url = f'{url}/invoke'
        print("REQUEST URL: ", request_url)
        response = httpx.post(url=request_url, json={"question": question}, timeout=timeout)
        print("STATUS CODE: ", response.status_code)
        print("TEXT: ", response.text)
        response.raise_for_status()
        return response.json()
    
    except httpx.TimeoutException as e:
        raise ClientError(f"Request time out after {timeout}s {e}")
    except httpx.ConnectError as e:
        raise ClientError(f'Connection Eorr on url: {url} {e}')
    except httpx.HTTPError as e:
        raise ClientError(f"Backend request fail {e}")

def upload_file(file: Any, timeout: float = 300, url: str= API_URL):
    try:
        upload_url = f'{url}/upload'
        print("UPLOAD FILE", upload_url)
        files={
            "file": (
                file.name,
                file.getvalue(),
                file.type
            )
        }
        response = httpx.post(
            url=upload_url, 
            files=files, 
            timeout=timeout
        )
        print("STATUS CODE: ", response.status_code)
        print("TEXT: ", response.text)
        response.raise_for_status()
        return response.json()
    
    except httpx.TimeoutException as e:
        raise ClientError(f"Request time out after {timeout}s {e}")
    except httpx.ConnectError as e:
        raise ClientError(f'Connection Eorr on url: {url} {e}')
    except httpx.HTTPError as e:
        raise ClientError(f"Backend request fail {e}")

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
        st.session_state.messages = [
            {
                "role": "ai",
                "message" : WELCOME
            }
        ]
        st.session_state.files = set()
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['message'])

    

    user_input = st.bottom.chat_input()

    uploaded_file = st.bottom.file_uploader("Choose a file", type=['pdf'])
    
    if user_input:
        st.session_state.messages.append(
            {
                "role": "human",
                "message" : user_input
            }
        )
        st.chat_message('human').write(user_input)
        with st.spinner("Fabricating..."):
            answer = invoke(user_input, 300)['answer']
        st.session_state.messages.append(
            {
                "role": "ai",
                "message" : answer
            }
        )
        st.chat_message('ai').write(answer)
        
    if uploaded_file is not None and uploaded_file.name not in st.session_state.files:

        result = upload_file(uploaded_file, timeout=300)
        if result['status'] == "ok":
            st.session_state.files.add(uploaded_file.name)
            st.success(f"Upload: {result['file_name']}")
            st.write(result)
        else:
            st.error(result['message'])
if __name__ == '__main__':
    main()
