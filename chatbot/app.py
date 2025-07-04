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

load_dotenv()
client = AsyncOpenAI()
cl.instrument_openai()

settings = {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 500}
base_url = "http://localhost:8001"

@cl.on_chat_start
async def start_chat():
    system_msg = "You are a drone flight data analyst. Answer questions accordingly."
    cl.user_session.set("chat_history", [{"role": "system", "content": system_msg}])

@cl.on_message
async def main(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    response = requests.get(url = f"{base_url}/api/files/")
    files = response.json()

    if files != []: 
        file_id = files[0]["file_id"]
        user_id = "fozyurt"
        response = requests.get(url = f"{base_url}/api/files/{file_id}/status", headers = {"user-id": user_id})
        flight_status = response.json()        
        
        input = f"""
                 Flight data is loaded:
                 File: {flight_status.get('filename')}clea
                 Content: {flight_status.get('content')}

                 User's query: {message.content}
                 """
        requests.delete(url = f"{base_url}/api/files/{file_id}", headers = {"user-id": user_id}) 
    else:
        input = message.content
    
    
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