from pymavlink import mavutil
import pandas as pd 
import gc
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import re
import requests
from langchain.chains import create_history_aware_retriever

def read_drone_log(file_path, msg_types):
    mlog = mavutil.mavlink_connection(file_path)
    log_data = {}
    message_count = 0
    try:
        while True:
            msg = mlog.recv_match(type=msg_types)
            if msg is None: 
                break
            msg_type = msg.get_type()
            
            if msg_type not in log_data: 
                log_data[msg_type] = []
                
            msg_dict = msg.to_dict()
            msg_dict['timestamp'] = msg._timestamp
            log_data[msg_type].append(msg_dict)
            message_count += 1
            
            if message_count % 10000 == 0:
                gc.collect()            
    finally:
        if hasattr(mlog, 'close'):
            mlog.close()        
            
    return log_data

def structure_log_data(log_data):
    structured_data = {}
    for msg_type, messages in log_data.items():
        if messages:  
            try:
                df = pd.DataFrame(messages)
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                structured_data[msg_type] = df
                messages.clear()
            except Exception as e:
                print(f"Error processing {msg_type}: {e}")
    gc.collect()
    return structured_data

def generate_gps_summary(file_path):
    log_data = read_drone_log(file_path, ["ATT", "GPS"])
    dfs = structure_log_data(log_data)
    gps_info = dfs["GPS"][:10].to_string().replace('  ', ' ')
    return gps_info

def get_flight_data(user_id, file_id):
    try:
        headers = {"user-id": user_id}
        response = requests.get(url = f"http://localhost:8001/api/files/{file_id}/status", headers = headers)
        return response.json()
    except:
        return {"has_file": False}

def extract_url(text):
    match = re.search(r'(https?://\S+)', text)
    return match.group(1) if match else None

def process_url(url, embedding_model, retrieval_llm):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever()
    qa_chain = create_history_aware_retriever(retrieval_llm, retriever, prompt)
    return qa_chain

# def get_answer(url):
#     loader = WebBaseLoader(url)
#     document = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(document)
#     vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

#     llm = ChatOpenAI()
#     retriever = vector_store.as_retriever()
#     prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"),
#                                                ("user", "{input}"),
#                                                ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")])
    
#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
#     llm = ChatOpenAI()
#     prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="chat_history"),
#                                                ("system", "Answer the user's questions based on the below context:\n\n{context}"),
#                                                ("user", "{input}"),])
    
#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
#     conversation_rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
#     response = conversation_rag_chain.invoke({"chat_history": st.session_state.chat_history,
#                                               "input": user_input})
    
#     return response['answer']