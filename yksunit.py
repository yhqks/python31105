
import chardet
import os
from typing import List, Union

# 所需依赖库（需提前安装）
# pip install python-docx python-pptx openpyxl pdfplumber Pillow

# ========== 模块导入 ==========
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from docx import Document
    from pptx import Presentation
    from openpyxl import load_workbook
    import pdfplumber
    from langchain_community.vectorstores import Chroma
    from openai import OpenAI
    import chardet
except ImportError:
    raise ImportError("请先安装依赖库：python-docx, python-pptx, openpyxl, pdfplumber,langchain_text_splitters,Chroma,openai,chardet")

# ========== 异常处理类 ==========
class DocumentParseError(Exception):
    """自定义文档解析异常"""
    def __init__(self, message: str):
        super().__init__(message)
        self.res()

    def res(self)->None:
        print("文档解析异常")
def save_vector(text: List[Document]) -> None:
    """
    保存向量
    :param text: 文档内容
    """
    client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
)
    completion = client.embeddings.create(
                    model="text-embedding-v3",
                    input=text
                    )
    # 创建向量存储
    vectorstore = Chroma(persist_directory='chroma_data_dir', embedding_function=completion)
    # 保存向量
    vectorstore.save(text)


def spilt_text(text:str)-> List[Document]:
    documents = [
    Document(
    page_content=str(text),
    metadata={"source": 'text'}
    )
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    res = splitter.split_documents(documents)
    return res
'''

'''
# ========== 核心解析器 ==========
class UniversalDocumentParser:
    """通用文档解析器"""
    def __init__(self):
        self.supported_formats = {
            '.docx': self._parse_word,
            '.pptx': self._parse_ppt,
            '.xlsx': self._parse_excel,
            '.csv': self._parse_excel,
            '.pdf': self._parse_pdf,
            '.doc': self._parse_word,  # 支持Word 97-2003文档
        }

    def parse(self, file_path: str) -> str:
        """
        通用文档解析入口
        :param file_path: 文件路径
        :return: 提取的文本内容
        """
        # 校验文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        # 获取文件扩展名
        ext = os.path.splitext(file_path)[1].lower()

        # 选择解析方法
        parser = self.supported_formats.get(ext)
        if not parser:
            raise DocumentParseError(f"不支持的文件格式：{ext}")

        try:
            return parser(file_path)
        except Exception as e:
            raise DocumentParseError(f"解析失败：{str(e)}") from e

    def _parse_word(self, file_path: str) -> str:
        """解析Word文档"""
        doc = Document(file_path)
        text = []
        
        # 提取段落文本
        for para in doc.paragraphs:
            text.append(para.text)
        
        # 提取表格内容
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text for cell in row.cells]
                text.append('\t'.join(row_text))
        
        return '\n'.join(text)

    def _parse_ppt(self, file_path: str) -> str:
        """解析PowerPoint文档"""
        prs = Presentation(file_path)
        text = []
        
        # 遍历所有幻灯片
        for slide in prs.slides:
            # 提取形状中的文本
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
                # 处理表格（PPTX中的表格）
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = [cell.text for cell in row.cells]
                        text.append('\t'.join(row_text))
        
        return '\n'.join(text)

    def _parse_excel(self, file_path: str) -> str:
        """解析Excel文档"""
        wb = load_workbook(file_path)
        text = []
        
        # 遍历所有工作表
        for sheet in wb:
            # 读取最大行和列
            max_row = sheet.max_row
            max_col = sheet.max_column
            
            # 逐行读取数据
            for row in sheet.iter_rows(max_row=max_row, max_col=max_col):
                row_text = [str(cell.value) if cell.value else "" for cell in row]
                text.append('\t'.join(row_text))
        
        return '\n'.join(text)

    def _parse_pdf(self, file_path: str) -> str:
        """解析PDF文档"""
        text = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # 提取文本
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                
                # 提取表格（需要处理表格结构）
                for table in page.extract_tables():
                    for row in table:
                        cleaned_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                        text.append('\t'.join(cleaned_row))
        
        return '\n'.join(text)



class FileEncoder:
    def __init__(self, input_file):
        """
        初始化 FileEncoder 类。
        
        :param input_file: 输入文件路径
        """
        self.input_file = input_file
        self.current_encoding = None

    def detect_encoding(self):
        """
        检测文件的当前编码格式。
        
        :return: 文件的编码格式
        """
        with open(self.input_file, "rb") as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            self.current_encoding = result["encoding"]
            return self.current_encoding

    def convert_encoding(self, output_file, target_encoding):
        """
        将文件从当前编码转换为目标编码。
        
        :param output_file: 输出文件路径
        :param target_encoding: 目标编码格式
        """
        if not self.current_encoding:
            self.detect_encoding()

        with open(self.input_file, "r", encoding=self.current_encoding, errors="replace") as file:
            content = file.read()

        with open(output_file, "w", encoding=target_encoding, errors="replace") as file:
            file.write(content)

        print(f"文件已从 {self.current_encoding} 转换为 {target_encoding}，并保存到 {output_file}")

    def convert_to_utf8(self, output_file):
        self.convert_encoding(output_file, "utf-8")

    def convert_to_gbk(self, output_file):
        self.convert_encoding(output_file, "gbk")

    def convert_to_latin1(self, output_file):
        self.convert_encoding(output_file, "latin1")

    def convert_to_windows1252(self, output_file):
        self.convert_encoding(output_file, "windows-1252")

    def convert_to_iso_8859_1(self, output_file):
        self.convert_encoding(output_file, "iso-8859-1")
# ========== 使用示例 ==========
if __name__ == "__main__":
    parser = UniversalDocumentParser()
    y=FileEncoder("C:/Users/yks/Desktop/AIOPS资料及代码/AIOPS资料及代码/代码及数据/隐患识别/tianjinxunjian\AIOps_ADBS\config\解决方案.csv")
    y.convert_to_iso_8859_1("C:/Users/yks/Desktop/AIOPS资料及代码/AIOPS资料及代码/代码及数据/隐患识别/tianjinxunjian\AIOps_ADBS\config\解决方案.csv")

    # # 示例文件路径
    # test_files = {
    #     "Word文档": "example.docx",
    #     "PPT文档": "ppt.ppt",
    #     "Excel文档": "data.xlsx",
    #     "PDF文档": "pdf.pdf"
    # }

    # for doc_type, path in test_files.items():
    #     try:
    #         content = parser.parse(path)
    #         print(f"========== {doc_type} 解析结果 ==========")
    #         print(content[:500] + "..." if len(content) > 500 else content)  # 截取前500字符
    #         print("\n")
    #     except Exception as e:
    #         print(f"解析 {doc_type} 失败：{str(e)}")