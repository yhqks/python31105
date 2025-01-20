import docx  
from sklearn.feature_extraction.text import TfidfVectorizer  

# 读取.docx文件  
def read_docx(file_path):  
    doc = docx.Document(file_path)  
    text = []  
    for para in doc.paragraphs:  
        text.append(para.text)  
    return ' '.join(text)  

# 确定docx文件的路径  
file_path = 'C:/Users/yks/Downloads/基于RFID的人体无器械活动识别研究.docx'  
text = read_docx(file_path)  

# 使用TF-IDF向量化文本  
vectorizer = TfidfVectorizer()  
X = vectorizer.fit_transform([text])  

# 输出向量  
print(X.toarray())  
print(X.shape)