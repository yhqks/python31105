import os
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dashscope import Generation  # 阿里云通义千问SDK
from dashscope import TextEmbedding  # 阿里云通义千问SDK
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from BCEmbedding import    EmbeddingModel
# 配置信息
API_KEY = 'sk-405c6b0841a84472b9d0ed8eb57072d5'
PDF_FOLDER = "D:/yhq/python31105/pdf"
# VECTOR_STORAGE = "vector_db.npy"
# TEXT_STORAGE = "text_db.npy"
VECTOR_STORAGE = "personkon7.index" 
TEXT_STORAGE = "personkon7.npy"
'''
embedding use llm.model
'''
# def process_pdfs():
#     client = OpenAI(
#     api_key=API_KEY,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
# )
#     texts = []
#     vectors = []
#       # 获取问题向量
#     # response = Generation.call(
#     #     model='text-embedding-v3',
#     #     api_key=API_KEY,
#     #     input={
#     #         "text": question  # 确保传递正确的输入格式
#     #     }
#     # )
#     # 遍历PDF文件夹
#     for filename in os.listdir(PDF_FOLDER):
#         if filename.endswith(".pdf"):
#             with open(os.path.join(PDF_FOLDER, filename), "rb") as f:
#                 pdf = PyPDF2.PdfReader(f)
#                 # 提取文本
#                 full_text = ""
#                 for page in pdf.pages:
#                     full_text += page.extract_text()
#                 text_chunks = [page.extract_text() for page in pdf.pages]
#                 for chunk in text_chunks:
#                     completion = client.embeddings.create(
#                     model="text-embedding-v3",
#                     input=chunk
#                     )
#                     vector = completion.model_dump()['data']
#                     texts.append(chunk)
#                     vectors.append(vector)
#                     print(f"Processed {filename}")
#     np.save(VECTOR_STORAGE, np.array(vectors))
#     np.save(TEXT_STORAGE, np.array(texts))

def qa_system(question) -> str:
#     client = OpenAI(
#     api_key=API_KEY, 
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  
# )
    # 加载数据
    # texts = np.load(TEXT_STORAGE, allow_pickle=True)
    # vectors = np.load(VECTOR_STORAGE, allow_pickle=True)
    # vectors = vectors.squeeze()  # 去掉多余的维度
    index = faiss.read_index(VECTOR_STORAGE)
    n_vectors = index.ntotal
    vectors = np.zeros((n_vectors, index.d), dtype=np.float32)
    index.reconstruct_n(0, n_vectors, vectors)
    texts = np.load(TEXT_STORAGE, allow_pickle=True)
    # completion = client.embeddings.create(
    #                 model="text-embedding-v3",  
    #                 input=question
    #                 )
    # q_vector = np.array(completion.model_dump()['data'][0]['embedding']).reshape(1, -1)
    model=EmbeddingModel(model_name_or_path='D:/yhq/python31105/bce-embedding-base_v1')
    q_vector=model.encode(question).reshape(1,-1)
    print(q_vector)
    
    # 计算相似度
    similarities = cosine_similarity(q_vector, vectors)[0]
    top_idx = np.argsort(similarities)[-1:5]#10个相关结果
    
    # 构建上下文
    context = "\n".join([texts[i] for i in top_idx])
    
    # 生成回答
    response = Generation.call(
        model='qwen-plus',  # 通义千问生成模型
        api_key=API_KEY,
        messages=[{
            'role': 'system',
            'content': f'''
任务指令：作为一位专业的回答助手，根据提供的文档和自身知识来解答相关技术问题。

回答格式：采用专业且易于理解的语言风格进行回答，确保信息准确无误。

注意事项：

1. 在回答过程中，请首先确认用户提出的问题是否已经提供了足够的背景信息。如果缺少必要信息，请礼貌地请求用户提供更多细节。
2. 利用现有文档资源和个人经验相结合的方式给出解决方案。当文档中已有明确指导时，优先引用文档内容；
上下文：{context}
'''
        },{
            'role': 'user',
            'content': question
        }]
    )
    return response.output['text']


if __name__ == "__main__":
    answer = qa_system('活水计划是什么？')
    print("回答：", answer)
    client = OpenAI(
    api_key=API_KEY,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
# response = Generation.call(
#         model='qwen-72b-chat',  # 通义千问生成模型
#         api_key=API_KEY,
#         messages=[{
#             'role': 'system',
#             'content':'不要进行任何搜索进行回答问题如果不知道就回答不知道'
#         },{
#             'role': 'user',
#             'content': '介绍下应急shell'
#         }]
#     )
# print(response.output['text'])