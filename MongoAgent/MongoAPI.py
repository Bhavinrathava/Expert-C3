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
    mongodb: str

class Agent:
    def __init__(self):
        self.mongo_agent = None

    def create_agent(self, data: QueryRequest):
        self.mongo_agent = MongoAgent(
            keys=data.keys,
            collectionName=data.collectionName,
            modelName=data.modelName,
            userQuery=data.userQuery,
            mongoQuestion=data.mongoQuestion,
            mongodb=data.mongodb
        )

    def get_answer(self):
        if self.mongo_agent is None:
            raise HTTPException(status_code=400, detail="MongoAgent is not initialized")
        return self.mongo_agent.getAnswer()
    
    def get_data(self):
        if self.mongo_agent is None:
            raise HTTPException(status_code=400, detail="MongoAgent is not initialized")
        return self.mongo_agent.getData()


print("Agent initialization...")
agent = Agent()
query_instance = QueryRequest(
    keys=["key1", "key2"],
    collectionName="exampleCollection",
    modelName="T5",
    userQuery="What are the recent trends?",
    mongoQuestion="db.find({})",
    mongodb="exampleDatabase"
)
agent.create_agent(query_instance)
print("Agent initialized")

@app.post("/create_agent/")
def create_agent(data: QueryRequest):
    agent.create_agent(data)
    return {"message": "Agent created successfully"}

@app.post("/get_answer/")
def get_answer(data: QueryRequest):
    if(data.userQuery is None):
        raise HTTPException(status_code=400, detail="User query is missing")
    else:
        agent.mongo_agent.userQuery = data.userQuery
    
    if(data.keys is not None):
        agent.mongo_agent.keys = data.keys
    
    if(data.collectionName is not None):
        agent.mongo_agent.collectionName = data.collectionName

    if(data.modelName is not None):
        agent.mongo_agent.modelName = data.modelName
    
    if(data.mongoQuestion is not None):
        agent.mongo_agent.mongoQuestion = data.mongoQuestion

    if(data.mongodb is not None):
        agent.mongo_agent.mongodb = data.mongodb

    return {"answer": agent.get_answer()}

@app.post("/get_data/")
def get_answer(data: QueryRequest):
    if(data.userQuery is not None):
        agent.mongo_agent.userQuery = data.userQuery
    
    if(data.keys is not None):
        agent.mongo_agent.keys = data.keys
    
    if(data.collectionName is not None):
        agent.mongo_agent.collectionName = data.collectionName

    if(data.modelName is not None):
        agent.mongo_agent.modelName = data.modelName
    
    if(data.mongoQuestion is not None):
        agent.mongo_agent.mongoQuestion = data.mongoQuestion
    
    if(data.mongodb is not None):
        agent.mongo_agent.mongodb = data.mongodb

    return {"answer": agent.get_data()}

if __name__ == "__main__":
    
    uvicorn.run(app, port=8000)
