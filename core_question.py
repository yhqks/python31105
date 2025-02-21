import requests
import json
import os
from openai import OpenAI
from dashscope import Generation 
import ast  
# 通义千问 API 的 URL 和 API Key（需要替换为实际的 API Key 和 URL）
API_URL =  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 示例 URL，请替换为实际的 API 地址
API_KEY = os.getenv('DASHSCOPE_API_KEY')  # 

client = OpenAI(
    api_key=API_KEY,  
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  
)
question = "打篮球"
response = Generation.call(
        model='qwen-72b-chat',  
        api_key=API_KEY,
        messages=[ {"role": "system", "content": 
                    '''
  作为专业的提取核心语言的智能体，你的职责是将复杂问题分解为一系列更小、更易于处理的子问题。再将这些问题重新组成一个句子使其简洁。
  如果遇到无法进一步拆分的问题或无法回答的情况，请直接回复原问题本身。

注意事项：
- 不需要对拆解过程或结果进行额外解释。
- 如果可以拆解，则仅返回结果本身。
- 若不能分解或无法回答，则直接返回用户提出的问题原文。

示例格式：

[
  "子问题1",
  "子问题2",
  ...
]->简洁的问题

请确保遵循上述指导原则来完成任务
'''
                  },
                  {"role": "user", "content": f"请将以下问题重组为简洁的问题：{question}"}]
    )
res=response.output['text']
print(res)
# list_data = ast.literal_eval(res)  
# for i  in list_data:
#     print(i)