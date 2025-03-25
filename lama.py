import ollama
import base64

# 读取图片并转换为 base64 格式
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 调用 Llama3.2-vision 模型
def analyze_image(image_path):
    # 将图片转换为 base64
    image_base64 = image_to_base64(image_path)

    # 调用模型
    response = ollama.generate(
        model='llama3.2-vision',  # 模型名称
        prompt=''' 识别图中的所有汉字。
        注意：只需要返回识别出来的汉字即可。不需要任何其他的内容包括提示。这点十分重要。
        '''
   ,  # 提示词
        images=[image_base64]  # 传入图片
    )

    # 返回模型的回复
    return response['response']

# 示例：识别图片
image_path = 'D:\yhq\python31105\技术改造.png'  # 替换为你的图片路径
result = analyze_image(image_path)
print("识别结果：")
print(result)