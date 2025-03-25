
import chardet
import os
from typing import List, Union
from typing import List, Tuple
import os
import tempfile
from docx import Document
from docx.image.image import Image
from pptx import Presentation
from openpyxl import load_workbook
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR
from collections import defaultdict
from docx.parts.image import ImagePart
# 所需依赖库（需提前安装）
# pip install python-docx python-pptx openpyxl pdfplumber Pillow

# ========== 模块导入 ==========
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from docx import Document
    from pptx import Presentation
    from openpyxl import load_workbook
    import pdfplumber
    # from langchain_community.vectorstores import Chroma
    # from openai import OpenAI
    import chardet
except ImportError:
    raise ImportError("请先安装依赖库：python-docx, python-pptx, openpyxl, pdfplumber,langchain_text_splitters,Chroma,openai,chardet")

# ========== 文本拆分函数 ==========
# def spilt_text(text:str)-> List[Document]:
#     documents = [
#     Document(
#     page_content=str(text),
#     metadata={"source": 'text'}
#     )
#     ]
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     res = splitter.split_documents(documents)
#     return res
# ========== 异常处理类 ==========
class DocumentParseError(Exception):
    """自定义文档解析异常"""
    def __init__(self, message: str):
        super().__init__(message)
        self.res()

    def res(self)->None:
        print("文档解析异常")
# def save_vector(text: List[Document]) -> None:
#     """
#     保存向量
#     :param text: 文档内容
#     """
#     client = OpenAI(
#     api_key=os.getenv('DASHSCOPE_API_KEY'),  
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
# )
    # completion = client.embeddings.create(
    #                 model="text-embedding-v3",
    #                 input=text
    #                 )
    # # 创建向量存储
    # vectorstore = Chroma(persist_directory='chroma_data_dir', embedding_function=completion)
    # # 保存向量
    # vectorstore.save(text)



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
        self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
            )
        self.temp_dir = tempfile.TemporaryDirectory()  # 用于暂存提取的图片
    def __del__(self):
        self.temp_dir.cleanup()  # 清理临时文件

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

    def _initialize_ocr(self):
        if self.ocr_model is None:
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
            )



    def _ocr_image(self, image_path: str) -> str:
        """执行OCR识别并返回文本"""
        try:
            result = self.ocr_model.ocr(image_path, cls=True)
            return ' '.join([word_info[1][0] for line in result for word_info in line])
        except Exception as e:
            return ''
        
    def _get_image_extension(self, part: ImagePart) -> str:
        """获取图片的扩展名"""
        # 根据图片的MIME类型获取扩展名
        mime_type = part.content_type
        if mime_type == "image/png":
            return "png"
        elif mime_type == "image/jpeg":
            return "jpg"
        elif mime_type == "image/gif":
            return "gif"
        elif mime_type == "image/bmp":
            return "bmp"
        elif mime_type == "image/tiff":
            return "tiff"
        else:
            return "bin"  # 默认扩展名
        
    def _extract_images_from_word(self, doc: Document) -> List[Tuple[str, str]]:
        """从Word文档提取图片并返回(图片路径, OCR文本)列表"""
        images = []
        # 遍历文档中的所有部分
        for part in doc.part.package.parts:
            # 检查是否是图片部分
            if isinstance(part, ImagePart):
                image_data = part._blob
                ext = self._get_image_extension(part)
                # 创建临时文件保存图片
                with tempfile.NamedTemporaryFile(
                    dir=self.temp_dir.name, 
                    suffix=f".{ext}", 
                    delete=False
                ) as f:
                    f.write(image_data)
                    image_path = f.name
                    images.append(image_path)
        # 对每张图片进行OCR识别
        return [(img, self._ocr_image(img)) for img in images]
    


    def _extract_images_from_ppt(self, prs: Presentation) -> List[Tuple[str, str]]:
        """从PPT提取图片"""
        images = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.shape_type == 13:  # 图片类型
                    image_data = shape.image.blob
                    ext = shape.image.ext
                    with tempfile.NamedTemporaryFile(
                        dir=self.temp_dir.name,
                        suffix=f".{ext}",
                        delete=False
                    ) as f:
                        f.write(image_data)
                        images.append(f.name)
        return [(img, self._ocr_image(img)) for img in images]
    

    def _extract_images_from_pdf(self, file_path: str) -> dict[int, list[tuple[str, str]]]:
        """按页码提取图片及其OCR结果"""
        page_images = defaultdict(list)
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if base_image["ext"] in ["png", "jpeg", "jpg"] and base_image.get("image"):
                    try:
                        with tempfile.NamedTemporaryFile(
                            dir=self.temp_dir.name,
                            suffix=f".{base_image['ext']}",
                            delete=False
                        ) as f:
                            f.write(base_image["image"])
                            img_path = f.name
                            # 直接执行OCR并存储结果
                            ocr_text = self._ocr_image(img_path) or ""
                            # 按页码分组存储 (PyMuPDF页码从0开始)
                            page_images[page_num].append((img_path, ocr_text))
                    except Exception as e:
                        print(f"图片处理失败 (页码{page_num}): {e}")
        return dict(page_images)
    

    def _parse_word(self, file_path: str) -> str:
        """解析Word文档，提取文字和图片中的文字"""
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

        # 提取图片并识别文字
        image_results = self._extract_images_from_word(doc)
        for img_path, ocr_text in image_results:
            text.append(ocr_text)
        self.__del__()
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
                if shape.shape_type == 13:  # 图片类型
                    image_data = shape.image.blob
                    ext = shape.image.ext
                    with tempfile.NamedTemporaryFile(
                        dir=self.temp_dir.name,
                        suffix=f".{ext}",
                        delete=False
                    ) as f:
                        f.write(image_data)
                        text.append(self._ocr_image(f.name))
        # image_results = self._extract_images_from_ppt(prs)
        # for img_path, ocr_text in image_results:
        #     text.append(ocr_text)
        self.__del__()
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
        """按页合并文本、表格和图片内容"""
        full_text = []
        # 先提取所有页面的图片信息 (页码 -> 图片列表)
        page_images = self._extract_images_from_pdf(file_path)

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_content = []
                
                # 1. 提取文本
                if (page_text := page.extract_text()):
                    page_content.append(page_text)
                
                # 2. 提取表格
                for table in page.extract_tables():
                    for row in table:
                        cleaned_row = [
                            str(cell).replace('\n', ' ') if cell else "" 
                            for cell in row
                        ]
                        page_content.append('\t'.join(cleaned_row))
                
                # 3. 添加本页图片OCR结果（pdfplumber和PyMuPDF页码都从0开始）
                if img_pairs := page_images.get(page_num):
                    for img_path, ocr_text in img_pairs:
                        page_content.append(ocr_text)
                
                # 合并本页所有内容
                if page_content:
                    full_text.append("\n".join(page_content))
        self.__del__()
        return "\n\n".join(full_text)


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
    text=parser.parse("C:/Users/yks/Desktop/AIOPS资料及代码/AIOPS资料及代码/资料/新建 Microsoft Word 文档.docx")
    print(text)
    # y=FileEncoder("C:/Users/yks/Desktop/AIOPS资料及代码/AIOPS资料及代码/代码及数据/隐患识别/tianjinxunjian\AIOps_ADBS\config\解决方案.csv")
    # y.convert_to_iso_8859_1("C:/Users/yks/Desktop/AIOPS资料及代码/AIOPS资料及代码/代码及数据/隐患识别/tianjinxunjian\AIOps_ADBS\config\解决方案.csv")

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