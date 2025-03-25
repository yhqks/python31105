from pdf2docx import Converter

def pdf_to_word(pdf_path, docx_path):
    try:
        # 创建转换器对象
        cv = Converter(pdf_path)
        # 转换全部页面
        cv.convert(docx_path)
        # 关闭转换器
        cv.close()
        print("转换成功！")
    except Exception as e:
        print(f"转换失败: {e}")

# 使用示例
pdf_file = "input.pdf"
word_file = "output.docx"
pdf_to_word(pdf_file, word_file)