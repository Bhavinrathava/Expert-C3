'''
Input : User Query 
Fetch : MOngo Schema 
Process : Find out the query to fill in mongo.find()
Process : call Mongo.find() with the query
Process : CLassify if the query result is enough to answer the question 
Process : If not, call the Web Search Agent 
Output : Return the result to the user
'''

def generate_query(userQuery):
    # Generate the query to be passed to the mongo.find()
    pass

def execute_query(query):
    # Execute the query on the mongo schema
    pass

def classify_result(result):
    # Classify if the result is enough to answer the question
    pass

def call_web_search_agent(userQuery):
    # Call the web search agent to fetch the answer
    pass


def generateAnswer(userQuery, result):
    # Generate the answer to be returned to the user
    pass

def MongoAgent(userQuery):
    query = generate_query(userQuery)
    result = execute_query(query)
    if classify_result(userQuery, result):
        return generateAnswer(userQuery, result)
    else:
        return call_web_search_agent(userQuery)