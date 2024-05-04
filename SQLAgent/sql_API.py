from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from sqlAgent import SQL_Agent

# Assume the run_inference function is imported here

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    metadata: str 

@app.post("/generate_query/")
async def generate_query(request: QueryRequest):
    try:
        agent = SQL_Agent(request.question, request.metadata)
        result = agent.run_inference()
        return {"query": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
