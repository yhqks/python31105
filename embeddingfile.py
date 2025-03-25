import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP库（不推荐长期使用）
import yksunit as yk
try:
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from BCEmbedding import EmbeddingModel
   import numpy as np
except ImportError:
   raise ImportError("请先安装依赖库：langchain_text_splitters,BCEmbedding,numpy")
model=EmbeddingModel(model_name_or_path='D:/yhq/python31105/bce-embedding-base_v1')
PDF_FOLDER = "D:/yhq/python31105/pdf"
VECTOR_STORAGE = "embedding_db2.npy"
TEXT_STORAGE = "textembedding_db2.npy"




# def fileEmbedding(file: str, VECTOR_STORAGE: str = "embedding_db2.npy", TEXT_STORAGE: str = "textembedding_db2.npy") -> None:
#     '''
#     file 文件夹路径或文件路径 
#     VECTOR_STORAGE 向量存储路径
#     TEXT_STORAGE 文本存储路径
#     '''
#     texts = []
#     vectors = []
    
#     if os.path.isfile(file):
#         try:
#             content = yk.UniversalDocumentParser.parse(file)
#             print(content[:500] + "..." if len(content) > 500 else content)  # 截取前500字符
#             print("\n")
#             split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#             res = split.split_text(content)
            
#             # 将每个文本块的嵌入向量添加到 vectors 中
#             for chunk in res:
#                 vectors.append(model.encode(chunk).reshape(1, -1))  # 确保每个向量的形状为 (1, n_features)
#                 texts.append(chunk)
#             print(f'解析完成文件：{file}')
#         except Exception as e:
#             raise Exception(f'解析失败：{e}')
#     else:
#         for path in os.listdir(file):
#             filepath = os.path.join(file, path)
#             try:
#                 content = yk.UniversalDocumentParser().parse(filepath)
#                 split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#                 res = split.split_text(content)
                
#                 # 将每个文本块的嵌入向量添加到 vectors 中
#                 for chunk in res:
#                     vectors.append(model.encode(chunk).reshape(1, -1))  # 确保每个向量的形状为 (1, n_features)
#                     texts.append(chunk)
#                 print(f'解析完成文件：{filepath}')
#             except Exception as e:
#                 print(f"解析失败 文件:{filepath} \n {e}")
    
#     # 将 vectors 和 texts 转换为 NumPy 数组
#     vectors = np.vstack(vectors)  # 使用 vstack 将列表中的数组堆叠成一个二维数组
#     texts = np.array(texts)
    
#     # 如果文件已存在，则加载现有数据并追加新数据
#     if os.path.exists(VECTOR_STORAGE) and os.path.exists(TEXT_STORAGE):
#         existing_vectors = np.load(VECTOR_STORAGE, allow_pickle=True)
#         existing_texts = np.load(TEXT_STORAGE, allow_pickle=True)
        
#         vectors = np.vstack([existing_vectors, vectors])  # 追加向量
#         texts = np.concatenate([existing_texts, texts])   # 追加文本
    
#     # 保存更新后的数据
#     np.save(VECTOR_STORAGE, vectors)
#     np.save(TEXT_STORAGE, texts)
def fileEmbedding(file: str, 
                 VECTOR_STORAGE: str = "embedding_db3.npy",
                 TEXT_STORAGE: str = "textembedding_db3.npy") -> None:
    '''
    递归处理文件夹和文件的嵌入生成函数
    file: 文件夹路径或文件路径 
    VECTOR_STORAGE: 向量存储路径
    TEXT_STORAGE: 文本存储路径
    '''
    texts = []
    vectors = []
    
    def process_file(file_path: str):
        """处理单个文件的内部函数"""
        try:
            content = yk.UniversalDocumentParser().parse(file_path)
            print(f"处理文件: {file_path}")
            print(content[:500] + "..." if len(content) > 500 else content)
            
            # 文本分割
            split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            res = split.split_text(content)
            
            # 生成嵌入
            for chunk in res:
                vectors.append(model.encode(chunk).reshape(1, -1))
                texts.append(chunk)
                
        except Exception as e:
            print(f"解析失败 {file_path} \n错误信息: {str(e)}")

    # 处理输入路径
    if os.path.isfile(file):
        process_file(file)
    elif os.path.isdir(file):
        # 递归遍历所有子目录
        for root, _, files in os.walk(file):
            for filename in files:
                file_path = os.path.join(root, filename)
                process_file(file_path)
    else:
        raise ValueError("输入路径既不是文件也不是目录")

    # 合并保存数据
    if vectors and texts:
        vectors_np = np.vstack(vectors)
        texts_np = np.array(texts)
        
        # 合并现有数据（如果存在）
        if os.path.exists(VECTOR_STORAGE) and os.path.exists(TEXT_STORAGE):
            existing_vectors = np.load(VECTOR_STORAGE, allow_pickle=True)
            existing_texts = np.load(TEXT_STORAGE, allow_pickle=True)
            
            vectors_np = np.vstack([existing_vectors, vectors_np])
            texts_np = np.concatenate([existing_texts, texts_np])
        
        # 保存结果
        np.save(VECTOR_STORAGE, vectors_np)
        np.save(TEXT_STORAGE, texts_np)
        print(f"数据已保存到 {VECTOR_STORAGE} 和 {TEXT_STORAGE}")
    else:
        print("未找到可处理的有效文件")
if __name__ == "__main__":
    fileEmbedding('D:/AIOPS资料及代码/AIOPS资料及代码/资料')
    print("处理完成")