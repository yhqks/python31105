from BCEmbedding import EmbeddingModel
sc=['这是个测试句子','这也是个测试句子']
model=EmbeddingModel(model_name_or_path='D:/yhq/python31105/bce-embedding-base_v1')
em=model.encode(sc)
print(em)