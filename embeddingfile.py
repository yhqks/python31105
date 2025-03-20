import os
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



def fileEmbedding(file: str, VECTOR_STORAGE: str = "embedding_db2.npy", TEXT_STORAGE: str = "textembedding_db2.npy") -> None:
    '''
    file 文件夹路径或文件路径 
    VECTOR_STORAGE 向量存储路径
    TEXT_STORAGE 文本存储路径
    '''
    texts = []
    vectors = []
    
    if os.path.isfile(file):
        try:
            content = yk.UniversalDocumentParser.parse(file)
            print(content[:500] + "..." if len(content) > 500 else content)  # 截取前500字符
            print("\n")
            split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            res = split.split_text(content)
            
            # 将每个文本块的嵌入向量添加到 vectors 中
            for chunk in res:
                vectors.append(model.encode(chunk).reshape(1, -1))  # 确保每个向量的形状为 (1, n_features)
                texts.append(chunk)
            print(f'解析完成文件：{file}')
        except Exception as e:
            raise Exception(f'解析失败：{e}')
    else:
        for path in os.listdir(file):
            filepath = os.path.join(file, path)
            try:
                content = yk.UniversalDocumentParser().parse(filepath)
                split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                res = split.split_text(content)
                
                # 将每个文本块的嵌入向量添加到 vectors 中
                for chunk in res:
                    vectors.append(model.encode(chunk).reshape(1, -1))  # 确保每个向量的形状为 (1, n_features)
                    texts.append(chunk)
                print(f'解析完成文件：{filepath}')
            except Exception as e:
                print(f"解析失败 文件:{filepath} \n {e}")
    
    # 将 vectors 和 texts 转换为 NumPy 数组
    vectors = np.vstack(vectors)  # 使用 vstack 将列表中的数组堆叠成一个二维数组
    texts = np.array(texts)
    
    # 如果文件已存在，则加载现有数据并追加新数据
    if os.path.exists(VECTOR_STORAGE) and os.path.exists(TEXT_STORAGE):
        existing_vectors = np.load(VECTOR_STORAGE, allow_pickle=True)
        existing_texts = np.load(TEXT_STORAGE, allow_pickle=True)
        
        vectors = np.vstack([existing_vectors, vectors])  # 追加向量
        texts = np.concatenate([existing_texts, texts])   # 追加文本
    
    # 保存更新后的数据
    np.save(VECTOR_STORAGE, vectors)
    np.save(TEXT_STORAGE, texts)

if __name__ == "__main__":
    fileEmbedding('D:/AIOPS资料及代码/AIOPS资料及代码/资料')
    print("处理完成")