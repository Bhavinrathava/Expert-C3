from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mongoAgent import MongoAgent
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    keys: list[str]
    collectionName: str
    modelName: str
    userQuery: str
    mongoQuestion: str

class Agent:
    def __init__(self):
        self.mongo_agent = None

    def create_agent(self, data: QueryRequest):
        self.mongo_agent = MongoAgent(
            keys=data.keys,
            collectionName=data.collectionName,
            modelName=data.modelName,
            userQuery=data.userQuery,
            mongoQuestion=data.mongoQuestion
        )

    def get_answer(self):
        if self.mongo_agent is None:
            raise HTTPException(status_code=400, detail="MongoAgent is not initialized")
        return self.mongo_agent.getAnswer()
    
    def get_data(self):
        if self.mongo_agent is None:
            raise HTTPException(status_code=400, detail="MongoAgent is not initialized")
        return self.mongo_agent.getData()

agent = Agent()

@app.post("/create_agent/")
def create_agent(data: QueryRequest):
    agent.create_agent(data)
    return {"message": "Agent created successfully"}

@app.get("/get_answer/")
def get_answer(question: str):
    agent.userQuery = question
    return {"answer": agent.get_answer()}

@app.get("/get_data/")
def get_answer():
    return {"answer": agent.get_data()}

if __name__ == "__main__":
    
    uvicorn.run(app, port=8000)
