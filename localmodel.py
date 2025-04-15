from openai import OpenAI

# 创建客户端并指向本地 Ollama 服务
client = OpenAI(
    base_url="113.89.32.168:40000/v1",  # Ollama 的 OpenAI 兼容端点
    api_key="ollama",  # 这里可以任意填写（Ollama 不验证 API key）
)

# 调用模型
response = client.chat.completions.create(
    model="deepseek-r1:70b",  # 你实际使用的模型名称（需与 Ollama 中名称一致）
    messages=[
        {"role": "system", "content": "你是一个有用的助手"},
        {"role": "user", "content": "讲一个关于人工智能的短故事"}
    ],
    temperature=0.7,
)

# 输出结果
print(response.choices[0].message.content)