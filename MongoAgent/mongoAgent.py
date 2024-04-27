
'''
Here we want to create a functionality that will allow us to convert user query into a mongo query.
'''
#from langchain.prompts import PromptTemplate
#from langchain_community.chat_models import ChatOllama
#from langchain_core.output_parsers import StrOutputParser

#import pymongo
import ollama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser



class MongoAgent:
    def __init__(self, ollamaModelName = "llama3", keys = None, collectionName = None, modelName = None, userQuery = None, mongoQuestion = None):
        self.keys = keys
        self.collectionName = collectionName
        self.modelName = modelName
        self.userQuery = userQuery
        self.mongoQuestion = mongoQuestion
        self.mongoQuery = None
        self.data = None
        self.answer = None
        self.ollamaModelName = ollamaModelName
        
        ollama.pull(ollamaModelName)

        self.model = AutoModelForSeq2SeqLM.from_pretrained("Chirayu/nl2mongo")
        self.tokeniezr = AutoTokenizer.from_pretrained("Chirayu/nl2mongo")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)


    def getAnswer(self):
        '''
        This function will call all the functions in the class and return the answer.
        '''
        
        self.build_mongo_query()
        self.execute_mongo_query()
        return self.build_answer()
    
    def build_mongo_query(self):
        input_ids = self.tokenizer.encode(
        self.mongoQuestion, return_tensors="pt", add_special_tokens=True
        )
        
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=10,
            max_length=128,
            repetition_penalty=2.5,
            length_penalty=1,
            early_stopping=True,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1,
        )

        query = [
            self.tokenizer.decode(
                generated_id,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for generated_id in generated_ids
        ][0]

        self.mongoQuery = query
        print("Mongo Query: ", query)
        return query

    def execute_mongo_query(self):
        '''
        This function will take in a mongo query and execute it.
        '''
        
        
        # Define connection to the database
        # Connect to the database
        # Select the collection 
        # Execute the query
        # Return the data
        self.data = "This is the data"

        return self.data

    def build_answer(self):
        '''
        This function will take in the data and the original query and build an answer.
        '''
        prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved data from MongoDB server to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {userQuery} 
        Context: {data} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
        )

        llm = ChatOllama(model=self.ollamaModelName, temperature=0)
        
        print("Initialising the chain...")
        rag_chain = prompt | llm | StrOutputParser()

        # Generate the answer
        self.answer = rag_chain.invoke({"question": self.userQuery, "context": (self.data)})
        
        print("Answer: ", self.answer)
        return self.answer

def main():
    print("Hello from MongoAgent ......")
    # Define the keys
    keys =["_id", "event_name", "event_date", "event_location", "event_category","event_description"]
    collectionName = "events"
    modelName = "T5"
    mongoQuestion = "What are the cultural events happening this week?"
    userQuery = "What are the cultural events happening this week?"

    agent = MongoAgent(keys = keys, collectionName = collectionName, modelName = modelName, userQuery = userQuery, mongoQuestion = mongoQuestion)
    answer = agent.getAnswer()

if __name__ == "__main__":
    main()
