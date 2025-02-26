import os
from pydantic import BaseModel
API_KEY = os.getenv("DASHSCOPE_API_KEY")
from dashscope import Generation
import json
from pprint import pprint
class bill(BaseModel):
    id:int
    Name:str
    charge:float
    totoal:float
    insurance:float
response = Generation.call(
        model='qwen-plus',  # 通义千问生成模型
        api_key=API_KEY,
        messages=[{
            'role': 'system',
            'content': f'''
任务指令：作为一个数据生成专家可以通过用户的输入的格式或者内容生成一些数据，数据内容进可能贴近真实。
注意：只需要返回生成的数据即可，甚至不需要输出返回的数据类型。并且数据应该与输入不同。
'''
        },{
            'role': 'user',
            'content': '''
class bill(BaseModel):
    id:int
    Name:str
    charge:float
    totoal:float
    insurance:float
这是要生成数据的类 其中id是整数 Name是人名 charge是费用 totoal是账单总数 insurance是保险能报销的费用
返回最起码10个符合这个类的数据 例如：
id:1
Name: 闫大帅
charge:1000
totoal:2000
insurance:500
返回格式要求：[id:1,Name:张三,charge:1000,totoal:2000,insurance:500]
其中数字可以随机生成 按照class里面的类型生成 其中charge要符合正态分布，并且每一个数据之间用，隔开生成内容里面不需要任何的换行符
'''
        }],
        result_format='messages'
    )
print(response.output['text'])
data_str = response.output['text'].replace('\n', '').replace(' ', '')

# 将字符串分割成多个记录
records = data_str.strip('[]').split('],[')

# 解析每条记录并保存为字典
data_list = []
for record in records:
    # 将记录转换为字典
    record_dict = {}
    for item in record.split(','):
        key, value = item.split(':')
        record_dict[key] = value
    data_list.append(record_dict)

# 打印结果
print(data_list)