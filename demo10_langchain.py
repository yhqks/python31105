"""

 Created on 2025/2/13  
 @author: yks 
 Path: D:/yhq/python31105/demo10_langchain.py
 use:llm vs rag
"""
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from fastapi import FastAPI
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

qianfan_ak = os.getenv("QIANFAN_AK")
qianfan_sk = os.getenv("QIANFAN_SK")
#得到模型：千帆
model = QianfanChatEndpoint(
    model="ERNIE-3.5-8K",
    temperature=0.2,
    timeout=60,
    api_key=qianfan_ak,
    secret_key=qianfan_sk,

)
'''

param=StrOutputParser()


#定义提示模板
prompt_template=ChatPromptTemplate.from_messages([
    ('human','请你作为一个大模型不进行任何检索回答下面的问题'),
    ('human','{text}')
])


#得到链
re=prompt_template|qianfan_chat|param
print(re.invoke({'text': '介绍下关于轻量化的RFID的识别技术'}))
'''
chroma_data_dir = 'chroma_data_dir'  # 替换为你的路径  
embedding_model = HuggingFaceBgeEmbeddings(api_key=qianfan_ak, secret_key=qianfan_sk)
vector_store = Chroma(persist_directory=chroma_data_dir,embedding_function=embedding_model) 
message = '''
请你作为一个大模型不进行任何检索回答下面的问题\n
{context}
'''
# 创建提示
prompt_template = ChatPromptTemplate.from_messages([
    ('human', message),
    ('human', "{input}")
])
#回答解析器
param = StrOutputParser()


retriever=vector_store.as_retriever()
# 创建检索链
chain = prompt_template|model | param
chain = create_retrieval_chain(retriever, chain)
print(chain.invoke({'input': '介绍下关于轻量化的RFID的识别技术'}))  