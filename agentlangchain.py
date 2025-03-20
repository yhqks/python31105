from json import load
from tabnanny import verbose
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.agent_toolkits.load_tools import load_tools

from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
serpapi=os.getenv('serpapi')
print(serpapi)
load_dotenv()
API_KEY=os.getenv("DASHSCOPE_API_KEY")
llm = ChatOpenAI(model='qwen-72b-chat', api_key=API_KEY, base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')

tools = load_tools(['serpapi','llm-math'],llm=llm,serpapi_api_key=serpapi)
memory=ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
)
agent=initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

print(agent.run("在纽约一百人民币能买多少支玫瑰"))