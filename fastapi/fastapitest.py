from fastapi import FastAPI
import uv
import uvicorn


app =  FastAPI(title="LangChain Server", version="1.0", description="A simple API server using LangChain's Runnable interfaces")
@app.get('/')


async def main():
    return {'message': 'Hello World'}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1',port=8000)