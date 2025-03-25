import torch
from paddleocr import PaddleOCR

# 检查PyTorch是否正常
print(torch.rand(2,3))  # 应输出随机矩阵

# 初始化PaddleOCR
ocr = PaddleOCR(use_angle_cls=True)  # 应无报错