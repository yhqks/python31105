from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import json
import asyncio
from langchain_openai import ChatOpenAI

prompt=ChatPromptTemplate.from_template('你是什么模型')
model=ChatOpenAI(model='deepseek-r1',api_key=os.getenv('DASHSCOPE_API_KEY'),base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
parser=StrOutputParser()

chain= prompt|model|parser

async def main():
    async for chunk in chain.astream({}):
        print(chunk,end='',flush=True)

asyncio.run(main())