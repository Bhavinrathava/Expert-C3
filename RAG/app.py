import chainlit as cl

### To run this:
#### 1. pip install chainlit pymilvus sentence-transformers openai
#### 2. open terminal, navigate to directory, enter: "chainlit run testBot.py -w"
#### 3. open http://localhost:8000/ 
from ragFlow import getAnswerWithRag

########################################
from openai import OpenAI
client = OpenAI(api_key="masked") #insert your own API key here

def llm_query(instructions):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are a policy expert"},
            {"role": "user", "content": instructions}]
    )
    response_content = response.choices[0].message.content
    print(response_content)
    return response_content
########################################

@cl.on_message
async def main(message: cl.Message):
    res = ask(message.content)
    
    # Send a response back to the user
    await cl.Message(
        content=res,
    ).send()


def classifier(query):
    return False #To be implemented

def rag_call(query):
    return getAnswerWithRag(query)

def post_process(context, query):
    return "Here is the user query:\n"+query+"\n### Here is helpful context:\n"+context #to be changed and improved

def ask(question):
    user_query = question
    if classifier(user_query):
        response = rag_call(user_query)
    else:
        response = llm_query(user_query)
    return response