from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
import tiktoken
from openai import OpenAI
llm=OpenAI(model='gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))

# 初始化 LLM 和 Memory
encoding = tiktoken.get_encoding("cl100k_base")  # 使用 GPT-3.5/GPT-4 的编码

# 定义 Token 阈值
MAX_TOKEN_LIMIT = 500  # 内存的最大 Token 限制
SUMMARY_TOKEN_LIMIT = 250  # 总结后的最大 Token 限制

def count_tokens(text):
    """统计文本的 Token 数量"""
    return len(encoding.encode(text))

def summarize_memory(memory):
    """总结内存中的对话历史"""
    buffer = memory.load_memory_variables({})["history"]
    
    # 调用 LLM 生成总结
    summary_prompt = ChatPromptTemplate.from_template("请总结以下对话:\n\n{conversation}\n\n总结:")
    summary_chain = summary_prompt | model | parser
    summary = summary_chain.invoke({"conversation": buffer})
    
    return summary

def update_memory_with_summary(memory, summary):
    """用总结内容更新内存"""
    memory.clear()
    memory.save_context({"input": "总结历史"}, {"output": summary})

def prune_memory_if_needed(memory):
    """如果内存内容过长，先总结，如果总结后仍然过长则截断"""
    buffer = memory.load_memory_variables({})["history"]
    token_count = count_tokens(buffer)
    
    if token_count > MAX_TOKEN_LIMIT:
        # 尝试总结
        summary = summarize_memory(memory)
        summary_token_count = count_tokens(summary)
        
        if summary_token_count <= SUMMARY_TOKEN_LIMIT:
            # 如果总结后的内容在限制内，更新内存
            update_memory_with_summary(memory, summary)
        else:
            # 如果总结后的内容仍然过长，截断内存
            messages = buffer.split("\n")
            while token_count > SUMMARY_TOKEN_LIMIT and len(messages) > 1:
                messages.pop(0)  # 移除最早的消息
                buffer = "\n".join(messages)
                token_count = count_tokens(buffer)
            
            # 更新内存
            memory.clear()
            memory.save_context({"input": "截断历史"}, {"output": buffer})

# 初始化模型和内存
model = ChatOpenAI(model='qwen-72b-chat', api_key=os.getenv('DASHSCOPE_API_KEY'), base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
memory = ConversationBufferMemory()

# 初始化 Prompt 和输出解析器
prompt = ChatPromptTemplate.from_template('{topic}')
parser = StrOutputParser()

# 创建链
chain = prompt | model | parser

async def chat_loop():
    while True:
        # 接受用户输入
        user_input = input("你: ")
        
        # 如果用户输入 "exit"，退出循环
        if user_input.lower() == "exit":
            print("对话结束。")
            break
        
        # 生成模型输出
        response = await chain.ainvoke({"topic": user_input})
        
        # 保存对话到内存
        memory.save_context({"input": user_input}, {"output": response})
        
        # 检查并处理内存
        prune_memory_if_needed(memory)
        
        # 打印模型输出
        print(f"AI: {response}")
        
        # 打印当前内存内容
        print("当前内存内容:", memory.load_memory_variables({}))

# 运行异步聊天循环
asyncio.run(chat_loop())