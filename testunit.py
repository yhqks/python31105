import yksunit


url='C:/Users/yks/Desktop/AIOPS资料及代码/AIOPS资料及代码/资料/基站全生命周期智能运维系统-结题汇报20240318.pptx'
parser = yksunit.UniversalDocumentParser()
print(parser.parse(url))