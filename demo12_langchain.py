import os
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 定义数据模型
class Classification(BaseModel):
    sentiment: str = Field(...,enum=['开心','伤心','难过','生气'],description='情感分类')
    aggressiveness: str = Field(description='描述文章的攻击性，越大表示攻击性越强，最高为10')
    language: str = Field(description='文本的语言')

# 初始化模型
model = ChatOpenAI(model='qwen-72b-chat', api_key=API_KEY, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')

# 定义提示模板
tagging_chat = ChatPromptTemplate.from_template(
    '''
    从下列段落中提取所需要的信息
    需要提取以下属性：
    - 情感分类 (sentiment) enum=['开心','伤心','难过','生气']
    - 攻击性 (aggressiveness, 范围 0-10)
    - 语言 (language)

    段落：
    {input}
    '''
)

# 输入文本
inputtext = '''我操你妈'''

# 调用模型并获取原始输出
raw_output = model.invoke(tagging_chat.format(input=inputtext))

# 打印原始输出
print("Raw Output:", raw_output)

# # 解析 raw_output 的内容
content = raw_output.content  # 提取 content 字段
print("Content:", content)

# 从 content 中提取所需信息
data = {}
for line in content.split('\n'):
    if '：' in line:  # 使用中文冒号分隔
        key, value = line.split('：', 1)
        key = key.strip()
        value = value.strip()
        if key == '情感分类':
            data['sentiment'] = value
        elif key == '攻击性':
            data['aggressiveness'] = str(value)
        elif key == '语言':
            data['language'] = value

# 将解析后的数据转换为 Classification 对象
try:
    result = Classification(**data)
    print("Parsed Result:", result)
except ValidationError as e:
    print("Validation Error:", e)