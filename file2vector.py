import os
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dashscope import Generation  # 阿里云通义千问SDK
from dashscope import TextEmbedding  # 阿里云通义千问SDK
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 配置信息
API_KEY = os.getenv("DASHSCOPE_API_KEY")
PDF_FOLDER = "D:/yhq/python31105/pdf"
VECTOR_STORAGE = "vector_db.npy"
TEXT_STORAGE = "text_db.npy"

def process_pdfs():
    client = OpenAI(
    api_key=API_KEY,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
    texts = []
    vectors = []
      # 获取问题向量
    # response = Generation.call(
    #     model='text-embedding-v3',
    #     api_key=API_KEY,
    #     input={
    #         "text": question  # 确保传递正确的输入格式
    #     }
    # )
    # 遍历PDF文件夹
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            with open(os.path.join(PDF_FOLDER, filename), "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                # 提取文本
                full_text = ""
                for page in pdf.pages:
                    full_text += page.extract_text()
                text_chunks = [page.extract_text() for page in pdf.pages]
                for chunk in text_chunks:
                    completion = client.embeddings.create(
                    model="text-embedding-v3",
                    input=chunk
                    )
                    vector = completion.model_dump()['data']
                    texts.append(chunk)
                    vectors.append(vector)
                    print(f"Processed {filename}")
    np.save(VECTOR_STORAGE, np.array(vectors))
    np.save(TEXT_STORAGE, np.array(texts))

def qa_system(question):
    client = OpenAI(
    api_key=API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  
)
    # 加载数据
    vectors = np.load(VECTOR_STORAGE, allow_pickle=True)
    texts = np.load(TEXT_STORAGE, allow_pickle=True)
    
  
    completion = client.embeddings.create(
                    model="text-embedding-v3",  
                    input=question
                    )
    q_vector = np.array(completion.model_dump()['data'][0]['embedding']).reshape(1, -1)
    print(q_vector)
    
    # 计算相似度
    similarities = cosine_similarity(q_vector, vectors)[0]
    top_idx = np.argsort(similarities)[-3:]  # 取前10个相关结果
    
    # 构建上下文
    context = "\n".join([texts[i] for i in top_idx])
    
    # 生成回答
    response = Generation.call(
        model='qwen-72b-chat',  # 通义千问生成模型
        api_key=API_KEY,
        messages=[{
            'role': 'system',
            'content': f'''
任务指令：作为一位专业的通信服务器运维工程师，根据提供的文档和自身专业知识来解答相关技术问题。

回答格式：采用专业且易于理解的语言风格进行回答，确保信息准确无误。

注意事项：

1. 在回答过程中，请首先确认用户提出的问题是否已经提供了足够的背景信息（如服务器型号、遇到的具体错误代码或现象等）。如果缺少必要信息，请礼貌地请求用户提供更多细节。
2. 利用现有文档资源和个人经验相结合的方式给出解决方案。当文档中已有明确指导时，优先引用文档内容；对于文档未覆盖的情况，则基于个人经验和最佳实践提供建议。
3. 保持答案结构清晰有序，可以按照“问题描述 -> 分析原因 -> 解决方案”的逻辑顺序组织语言。
4. 如果存在多种可能的解决方法，请列出所有选项，并简要说明每种方法的优点与局限性，以便用户根据实际情况选择最合适的方法。 
5. 若用户提出的问题无法回答，请礼貌地说明原因，并鼓励用户提供更多信息以便更好地帮助他们。
6. 请确保回答内容专业、准确、易懂，避免使用不当语言或行为。
上下文：{context}
'''
        },{
            'role': 'user',
            'content': question
        }]
    )
    return response.output['text']


if __name__ == "__main__":
        # answer = qa_system('display命令如何显示行数？')
        # print("回答：", answer)
    client = OpenAI(
    api_key=API_KEY,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
)
response = Generation.call(
        model='qwen-max',  # 通义千问生成模型
        api_key=API_KEY,
        messages=[{
            'role': 'system',
            'content':'不要进行任何搜索进行回答问题如果不知道就回答不知道'
        },{
            'role': 'user',
            'content': '新华3的路由设备的CLI界面display 查看信息时如何查看行号'
        }]
    )
print(response.output['text'])