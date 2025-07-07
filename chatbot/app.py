from openai import AsyncOpenAI
import chainlit as cl
import requests
from process import *
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import MessagesPlaceholder
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.docstore.document import Document
import os 
from chainlit.types import ThreadDict

load_dotenv()
client = AsyncOpenAI()
cl.instrument_openai()

settings = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 2000}
base_url = os.getenv("API_BASE_URL")
index_path = "faiss_index"
embedding_model = OpenAIEmbeddings()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    else:
        return None

@cl.on_chat_start
async def start_chat():
    system_msg = "You are a drone flight data analyst. Answer questions accordingly."
    cl.user_session.set("chat_history", [{"role": "system", "content": system_msg}])

@cl.on_message
async def main(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    # Get the right file for the current user. What if another user submits another file later? Will it be used for this user?
    response = requests.get(url = f"{base_url}/api/files/")
    files = response.json()

    if files != []: 
        file_id = files[0]["file_id"]
        user_id = "fozyurt"
        response = requests.get(url = f"{base_url}/api/files/{file_id}/status", headers = {"user-id": user_id})
        flight_status = response.json()             
        
        input = f"""
                 Flight data is loaded:
                 File: {flight_status.get('filename')}
                 Content: {flight_status.get('content')}
                 User's query: {message.content}
                 """
        # Put this to API
        requests.delete(url = f"{base_url}/api/files/{file_id}", headers = {"user-id": user_id}) 
    else:
        input = message.content
        
    vectorstore_body = {"content": message.content, "index_path": index_path}     
    status = requests.post(f"{base_url}/api/vectorstore/update", json = vectorstore_body).json().get("status", "")
    retrieved_context = requests.post(f"{base_url}/api/vectorstore/query", json = vectorstore_body).json().get("context", "")
    
    if retrieved_context:
        input += f"\nRetrieved context: {retrieved_context}"
        
    prompt_template = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}")])  
    prompts = prompt_template.format_messages(chat_history = chat_history, input = input)
    chat_history.append({"role": "user", "content": input})
    
    openai_messages = [{"role": convert_role(prompt.type), "content": prompt.content} for prompt in prompts]    
    stream = await client.chat.completions.create(messages = openai_messages,
                                                  stream = True,
                                                  # user = 'fozyurt',
                                                  **settings)
    msg = cl.Message(content = "")
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)
    
    chat_history.append({"role": "assistant", "content": msg.content})

    cl.user_session.set("chat_history", chat_history)
    await msg.update()
    
@cl.on_stop
def on_stop():
    print("The user wants to stop the task!")
    
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")
    
@cl.on_chat_end
def on_chat_end():
    print("The user disconnected!")