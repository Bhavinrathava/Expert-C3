import requests
# TODO - assume your inputs are current query from user and chat history 
# TODO - Send the question to the query_breakdown module and get the output in format 
# [{"Query":"Query Text", "Domain":"domain Name"},....]
# TODO - Iterate over the output from the step 2 and send the queries Async to available domains
    # if domain == "something" call func X with query
    # if domain == "something_else" call func Y with query

# TODO - Get the outputs from the various func calls -> collate and generate the final answer 

from Query_Breakdown import query_breakdown 
from RAG.ragFlow import getAnswerWithRag
import os 

def generate_final_answers(query_domain_mapping):
    final_answer = ""
    # do somethhing 
    return final_answer

def get_answer(user_query, chat_history):
    query_domain_mapping = query_breakdown(user_query)
    # Need to do this async 
    for query_domain in query_domain_mapping:
        if query_domain["Domain"] == "NUPolicy":
            # send query to NUPolicy and get data for the question
            # append the data from the domain query call as answer to the original dictionary
            query_domain["answer"] = getAnswerWithRag(query_domain_mapping)
            
        
        elif query_domain["Domain"] == "Events":
            if(os.environ.get("MONGO_API_URI") == None):
                raise ValueError("Please set the MONGO_API_URI environment variable.")
            
            # send the query to mongo seq2sql and get data for the question 
            base_url = os.environ.get("MONGO_API_URI")
            data = {
                # Title, day, time, desc, location, contact, reg_link
                "keys": ['Title', 'Day', 'Time', 'Desc', 'Location', 'Contact', 'Reg_link'],
                "collectionName": "events",
                "modelName": "T5",
                "userQuery": user_query,
                "mongoQuestion": query_domain["query"]
            }
            
            answerJSON = requests.post(base_url+"get_data/", data=data)

            if(answerJSON.status_code == 200):
                query_domain["answer"] = answerJSON.json()["answer"]
            else:
                print("Error in getting data from MongoAgent")
                query_domain["answer"] = "Error in getting data from MongoAgent"
        
    
    return generate_final_answers(query_domain_mapping)