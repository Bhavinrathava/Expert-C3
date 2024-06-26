
'''
Here we want to create a functionality that will allow us to convert user query into a mongo query.
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from functools import lru_cache
import re
#import pymongo
import ollama
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama

from nl2query import MongoQuery
from pymongo import MongoClient # import if performing analysis using python client
import json 
class MongoAgent:
    def __init__(self, mongodb = "yelp", ollamaModelName = "llama3", keys = None, collectionName = None, modelName = "T5", userQuery = None, mongoQuestion = None):
        self.keys = keys
        self.collectionName = collectionName
        self.modelName = modelName
        self.userQuery = userQuery
        self.mongoQuestion = mongoQuestion
        self.mongoQuery = None
        self.data = None
        self.answer = None
        self.ollamaModelName = ollamaModelName
        self.mongodb = mongodb
        #ollama.pull(ollamaModelName)

        self.model = AutoModelForSeq2SeqLM.from_pretrained("Chirayu/nl2mongo")
        self.tokenizer = AutoTokenizer.from_pretrained("Chirayu/nl2mongo")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @lru_cache(maxsize=100)
    def getAnswer(self):
        '''
        This function will call all the functions in the class and return the answer.
        '''
        
        self.build_mongo_query()
        self.execute_mongo_query()
        return self.build_answer()
    
    def build_mongo_query(self):
        keys = self.keys
        collectionName = self.collectionName
        modelName = self.modelName
        mongoQuestion = self.mongoQuestion
        queryfier = MongoQuery(model_type=modelName, collection_keys = keys, collection_name = collectionName)
        query = queryfier.generate_query(mongoQuestion)
        print("Query: ", query)
        self.mongoQuery = query

    def execute_mongo_query(self):
        '''
        This function will take in a mongo query and execute it.
        '''
        # Define a regex pattern that captures find, projection, sort, and limit from the MongoDB query string
        regex_pattern = (
            r"db\.(\w+)\.find\((\{.*?\}),\s*(\{.*?\})\)"  # captures collection, find dictionary, and projection dictionary
            r"(?:\.sort\((\{.*?\})\))?"                  # optionally captures sort dictionary
            r"(?:\.limit\((\d+)\))?"                     # optionally captures limit number
        )

        # Match the regex pattern against the query string
        match = re.search(regex_pattern, self.mongoQuery)
        if not match:
            raise ValueError("Query format is incorrect or not supported.")

        # Extract parts from the match
        collection_name, find_str, projection_str, sort_str, limit_str = match.groups()

        # Convert string representations to actual Python dict or int, using json.loads for safe parsing
        find_dict = json.loads(find_str.replace("'", '"')) if find_str else {}
        projection_dict = json.loads(projection_str.replace("'", '"')) if projection_str else {}
        sort_dict = json.loads(sort_str.replace("'", '"')) if sort_str else {}
        limit_num = int(limit_str) if limit_str else None

        for key in find_dict:
            # Capitalize the values of the keys
            find_dict[key] = find_dict[key].capitalize()

        # Executing the parsed query
        assert os.environ.get("MONGO_URI") is not None, "Please set the MONGO_URI environment variable."
        assert os.environ.get("MONGO_CONFIG_PATH") is not None, "Please set the MONGO_CONFIG_PATH environment variable."
        uri = os.environ.get("MONGO_URI")

        client = MongoClient(uri,
                        tls=True,
                        tlsCertificateKeyFile=os.environ.get("MONGO_CONFIG_PATH"))
        
        db = client[self.mongodb]
        collection = db[self.collectionName]
        print("Collection: ", collection)
        print("Find Dict: ", find_dict)
        print("Projection Dict: ", projection_dict)

        query_result = collection.find(find_dict, {'_id':0})
        self.data = list(query_result)
        print("Data: ", self.data)
        return self.data

    def getData(self):
        print(f"Building Mongo Query with the question: {self.mongoQuestion} \n")
        self.build_mongo_query()
        print(f'Generated MongoDB query: {self.mongoQuery} \n')

        print("Executing Mongo Query...")
        self.execute_mongo_query()
        return self.data
    
    def build_answer(self):
        '''
        This function will take in the data and the original query and build an answer.
        '''
        prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved data from MongoDB server to answer the question. If you don't know the answer, just say that you don't know. 
        Keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {userQuery} 
        Context: {data} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
        )
        
        assert os.environ.get("OLLAMA_BASE_URL") is not None, "Please set the OLLAMA_BASE_URL environment variable."

        llm = ChatOllama(model=self.ollamaModelName, base_url=os.environ.get("OLLAMA_BASE_URL"),temperature=0)
        
        print("Initialising the chain...")
        rag_chain = prompt | llm | StrOutputParser()

        # Generate the answer
        self.answer = rag_chain.invoke({"userQuery": self.userQuery, "data": (self.data)})
        
        print("Answer: ", self.answer)
        return self.answer

def main():
    print("Hello from MongoAgent ......")
    # Define the keys
    keys =['_id', 'index', 'business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'is_open', 'categories']
    collectionName = "Business"
    modelName = "T5"
    mongoQuestion = "What are the businesses in the city of Affton?"
    userQuery = "What are the businesses in the city of Affton?"

    agent = MongoAgent(keys = keys, collectionName = collectionName, modelName = modelName, userQuery = userQuery, mongoQuestion = mongoQuestion)
    answer = agent.getAnswer()
    print("Answer: ", answer)


if __name__ == "__main__":
    main()
