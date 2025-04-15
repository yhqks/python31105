# -*- coding: utf-8 -*-
import os

import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import NLTKTextSplitter
import yksunit as yk
try:
   from langchain_text_splitters import RecursiveCharacterTextSplitter
   from BCEmbedding import EmbeddingModel
   import numpy as np
   import faiss
except ImportError:
   raise ImportError("请先安装依赖库：langchain_text_splitters,BCEmbedding,numpy")
model=EmbeddingModel(model_name_or_path='D:/yhq/python31105/bce-embedding-base_v1')
PDF_FOLDER = "D:/yhq/python31105/pdf"
VECTOR_STORAGE = "personkon7.index" 
TEXT_STORAGE = "personkon7.npy"
failfile=[]


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
                 VECTOR_STORAGE: str = VECTOR_STORAGE,
                 TEXT_STORAGE: str = TEXT_STORAGE) -> None:
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
            text_splitter = NLTKTextSplitter()
            # split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            res = text_splitter.split_text(content)
            
            # 生成嵌入
            for chunk in res:
                vectors.append(model.encode(chunk).reshape(1, -1))
                texts.append(chunk)
                
        except Exception as e:
            failfile.append(file_path)
            # 处理异常，记录失败的文件路径和错误信息
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
        vectors_np = np.vstack(vectors)  # 向量数据
        texts_np = np.array(texts,dtype=object)
     

    # 检查是否需要合并现有数据
        assert len(vectors) > 0 and len(texts) > 0, "输入数据不能为空"
        vectors_np = np.vstack(vectors)  # 向量数据
        texts_np = np.array(texts, dtype=object)  # 文本数据

        # 检查向量和文本数量是否匹配
        assert vectors_np.shape[0] == len(texts_np), "向量数与文本数不匹配"

        # 处理向量索引
        if os.path.exists(VECTOR_STORAGE):
            try:
                existing_index = faiss.read_index(VECTOR_STORAGE)
                d = existing_index.d  # 正确获取维度
                existing_vectors = existing_index.ntotal
                
                # 提取现有向量
                existing_vectors_np = np.zeros((existing_vectors, d), dtype=np.float32)
                existing_index.reconstruct_n(0, existing_vectors, existing_vectors_np)
                
                # 检查新旧向量维度是否一致
                assert d == vectors_np.shape[1], f"维度不匹配：现有{d}维，新数据{vectors_np.shape[1]}维"
                
                # 合并向量
                all_vectors = np.vstack([existing_vectors_np, vectors_np])
            except Exception as e:
                print(f"加载现有索引失败，将重建索引。错误：{e}")
                all_vectors = vectors_np
        else:
            all_vectors = vectors_np

        # 创建新索引（统一用FlatL2简化流程）
        d = all_vectors.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(all_vectors)

        # 保存索引
        faiss.write_index(index, VECTOR_STORAGE)
        print(f"向量索引已保存到 {VECTOR_STORAGE}")

    # 处理文本数据
    if os.path.exists(TEXT_STORAGE):
        try:
            existing_texts = np.load(TEXT_STORAGE, allow_pickle=True)
            # 检查现有文本与向量数量是否匹配
            if len(existing_texts) != existing_index.ntotal:  # 仅在加载成功时检查
                raise ValueError(f"文本与向量数量不匹配：文本{len(existing_texts)}条，向量{existing_index.ntotal}条")
            texts_np = np.concatenate([existing_texts, texts_np])
        except Exception as e:
            print(f"加载现有文本失败，将覆盖保存。错误：{e}")

    # 最终检查一致性
    assert len(texts_np) == index.ntotal, "保存前检查失败：文本与向量数量不一致"
    np.save(TEXT_STORAGE, texts_np)
    print(f"文本数据已保存到 {TEXT_STORAGE}")

#     def fileEmbedding(file: str, 
#                  VECTOR_STORAGE: str = VECTOR_STORAGE,
#                  TEXT_STORAGE: str = TEXT_STORAGE,
#                  max_workers: int = os.cpu_count()*2) -> None:
#     '''
#     多线程版文档向量化函数
#     max_workers: 线程池大小 (建议设置为CPU核心数的2-3倍)
#     '''
#     # 线程安全的数据结构
#     text_queue = queue.Queue()
#     vector_queue = queue.Queue()
#     error_queue = queue.Queue()

#     # 文件发现线程
#     def file_discoverer(root_path: str):
#         """生产者线程：递归查找所有文件"""
#         try:
#             if os.path.isfile(root_path):
#                 file_queue.put(root_path)
#             elif os.path.isdir(root_path):
#                 for current_dir, _, files in os.walk(root_path):
#                     for filename in files:
#                         file_path = os.path.join(current_dir, filename)
#                         file_queue.put(file_path)
#             file_queue.put(None)  # 结束信号
#         except Exception as e:
#             error_queue.put(f"目录遍历错误: {str(e)}")

#     # 文件处理线程
#     def file_processor():
#         """消费者线程：处理单个文件"""
#         while True:
#             file_path = file_queue.get()
#             if file_path is None:  # 终止信号
#                 file_queue.put(None)  # 传递终止信号
#                 break
            
#             try:
#                 content = yk.UniversalDocumentParser().parse(file_path)
#                 splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=500, 
#                     chunk_overlap=50
#                 )
#                 chunks = splitter.split_text(content)
                
#                 for chunk in chunks:
#                     vector = model.encode(chunk).reshape(1, -1)
#                     text_queue.put(chunk)
#                     vector_queue.put(vector)
                
#             except Exception as e:
#                 error_msg = f"文件处理失败 {file_path}: {str(e)}"
#                 error_queue.put(error_msg)

#     # 启动线程池
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # 初始化任务队列
#         file_queue = queue.Queue()
        
#         # 启动文件发现线程
#         executor.submit(file_discoverer, file)
        
#         # 启动处理线程池
#         processor_futures = [
#             executor.submit(file_processor)
#             for _ in range(max_workers)
#         ]

#         # 实时数据保存线程
#         def save_worker():
#             """异步保存线程"""
#             batch_size = 1000  # 批处理大小
#             text_buffer = []
#             vector_buffer = []
            
#             while True:
#                 try:
#                     text = text_queue.get(timeout=5)
#                     vector = vector_queue.get(timeout=5)
                    
#                     # 批量保存
#                     if len(text_buffer) >= batch_size:
#                         _save_batch(text_buffer, vector_buffer)
#                         text_buffer.clear()
#                         vector_buffer.clear()
                        
#                 except queue.Empty:
#                     if all(f.done() for f in processor_futures):
#                         if text_buffer:  # 保存剩余数据
#                             _save_batch(text_buffer, vector_buffer)
#                         break

#         executor.submit(save_worker)

#     # 错误处理
#     while not error_queue.empty():
#         print(f"处理错误: {error_queue.get()}")

#     print("处理完成")

# def _save_batch(texts: list, vectors: list, 
#                VECTOR_STORAGE: str, TEXT_STORAGE: str):
#     """批量保存辅助函数"""
#     vectors_np = np.vstack(vectors)
#     texts_np = np.array(texts)
    
#     # 合并现有数据
#     if os.path.exists(VECTOR_STORAGE):
#         existing_vectors = np.load(VECTOR_STORAGE)
#         vectors_np = np.vstack([existing_vectors, vectors_np])
    
#     if os.path.exists(TEXT_STORAGE):
#         existing_texts = np.load(TEXT_STORAGE)
#         texts_np = np.concatenate([existing_texts, texts_np])
    
#     # 原子性保存
#     np.save(VECTOR_STORAGE+".tmp", vectors_np)
#     np.save(TEXT_STORAGE+".tmp", texts_np)
#     os.rename(VECTOR_STORAGE+".tmp", VECTOR_STORAGE)
#     os.rename(TEXT_STORAGE+".tmp", TEXT_STORAGE)
if __name__ == "__main__":
    start_time = time.time()
    fileEmbedding('E:/设计院2024年人才交流锻炼“活水20计划”.pdf',VECTOR_STORAGE,TEXT_STORAGE)
    end_time = time.time()
    print(f"处理完成，耗时 {end_time - start_time:.2f} 秒")
    print(f"失败文件列表：{failfile}")