import pandas as pd 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import re
from pymavlink import mavutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json

executor = ThreadPoolExecutor(max_workers=4)

def read_data(file_path, msg_types):    
    try:
        mlog = mavutil.mavlink_connection(file_path)
        data_collectors = defaultdict(list)
        
        while True:
            msg = mlog.recv_match(type = msg_types)
            if msg is None:
                break
                
            msg_type = msg.get_type()
            data_collectors[msg_type].append(msg.to_dict())
            
        return json.dumps(data_collectors["GPS"])
        
    except Exception as e:
        print(f"Error reading MAVLink file: {str(e)}")
        return ""
    
def convert_role(langchain_role):
    role_mapping = {"human": "user",
                    "ai": "assistant", 
                    "system": "system",
                    "user": "user",  
                    "assistant": "assistant" }
    
    return role_mapping.get(langchain_role, "user")  