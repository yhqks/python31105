# Description: This is a demo for using langchain to get the video information from youtube and then use the model to generate the subtitle.
from langchain_community.chat_models import QianfanChatEndpoint
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
qianfan_ak = os.getenv("QIANFAN_AK")
qianfan_sk = os.getenv("QIANFAN_SK")
embedding_model = QianfanEmbeddingsEndpoint(api_key=qianfan_ak, secret_key=qianfan_sk)
print(qianfan_sk,qianfan_ak)
# 得到模型：千帆
model = QianfanChatEndpoint(
    model="ERNIE-4.0-8K",
    temperature=0.2,
    timeout=180,
    api_key=qianfan_ak,
    secret_key=qianfan_sk,
)
#向量数据库路径
persist_dir='chroma_data_dir'
#获取向量
vectorstore=Chroma(persist_directory='chroma_data_dir',embedding_function=embedding_model)
#搜索
result=vectorstore.similarity_search_with_score('RFID ')

for res in result:
    print(res)
