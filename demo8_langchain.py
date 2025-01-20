import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
import docx
from langchain_core.documents import Document
# Step 1: Data Collection
file_path = 'C:/Users/yks/Downloads/基于RFID的人体无器械活动识别研究.docx'  # Replace with your local file path

def load_docx_file(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

docs=load_docx_file(file_path)
print(len(docs))
documents = [
Document(
    page_content=docs,
    metadata={"source": 'text'}
)
]
# Step 2: Data Processing
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
res = splitter.split_documents(documents)

# Step 3: Vectorization
qianfan_ak = os.getenv("QIANFAN_AK")
qianfan_sk = os.getenv("QIANFAN_SK")
embedding_model = QianfanEmbeddingsEndpoint(api_key=qianfan_ak, secret_key=qianfan_sk)
vectorstore = Chroma.from_documents(documents=res, embedding=embedding_model)

# Step 4: Storage
retriever = vectorstore.as_retriever()

# Step 5: Retrieval
model = QianfanChatEndpoint(
    model="ERNIE-4.0-8K",
    temperature=0.2,
    timeout=180,
    api_key=qianfan_ak,
    secret_key=qianfan_sk,
)

message = '''
你是一个助手，可以来回答关于文章内的问题,\n
{context}
'''
prompt_template = ChatPromptTemplate.from_messages([
    ('human', message),
    MessagesPlaceholder('chat_history'),
    ('human', "{input}")
])

param = StrOutputParser()
model = prompt_template|model | param
chain = create_retrieval_chain(retriever, model)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

result_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history',
    output_messages_key='answer'
)

# Example query
input_text = "这篇文章总共多少字 作者信息是什么"
resp = result_chain.invoke(
    {'input': input_text},
    config={'configurable': {'session_id': 'yks'}}
)
print(resp['answer'])