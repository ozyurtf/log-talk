from openai import AsyncOpenAI
import chainlit as cl
import requests
from process import *

client = AsyncOpenAI()
embedding_model = OpenAIEmbeddings()
retrieval_llm = ChatOpenAI()
cl.instrument_openai()

settings = {"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 500}

@cl.on_chat_start
def start_chat():
    system_msg = "You are a drone flight data analyst. Answer questions about uploaded flight logs."
    cl.user_session.set("message_history", [{"role": "system", "content": system_msg}])

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    response = requests.get(url = "http://localhost:8001/api/files/")
    files = response.json()

    if files != []: 
        file_id = files[0]["file_id"]
        user_id = "fozyurt"
        flight_status = get_flight_data(user_id, file_id)       
        
        context = f"""
                   FLIGHT DATA LOADED:
                   File: {flight_status.get('filename')}
                   Summary: {flight_status.get('summary')}

                   User's query: {message.content}
                   """
        headers = {"user-id": user_id}
        requests.delete(url = "http://localhost:8001/api/files/{file_id}", headers = headers)
    else:
        context = message.content
    
    url = extract_url(message.content)
    if url:
        try: 
            retriever = await process_url(url)
            cl.user_session.set("retriever", retriever)
        except Exception: 
            pass
    
    
    message_history.append({"role": "user", "content": context})    
    msg = cl.Message(content = result)
    stream = await client.chat.completions.create(messages = message_history, 
                                                  stream = True, 
                                                  # user = 'ubas213',
                                                  **settings)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    
    
@cl.on_message
async def on_message(message: cl.Message):
    url = extract_url(message.content)
    if url:
        qa = process_url(url)
        cl.user_session.set("qa", qa)

    qa = cl.user_session.get("qa")
    if qa:
        result = qa.run(message.content)
    else:
        result = ChatOpenAI().run(message.content) 

    await cl.Message(content=result).send()