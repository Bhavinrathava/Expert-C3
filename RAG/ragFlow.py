from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from openai import OpenAI

'''
Use the following code to get the answer from RAG

from ragFlow import getAnswerWithRag

query = "What is northwestern University's policy on Gifts?"
answer = getAnswerWithRag(query)
print(answer)

'''
def getEmbedding(query):
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #model = SentenceTransformer('all-mpnet-base-v2')
    return model.encode(query)

def getRelevantDocuments(query_embedding):
    client = MilvusClient(uri="https://in03-5c1e302a110bed7.api.gcp-us-west1.zillizcloud.com", token="db17b553ba29a8538ddaf9afa09396c271f9a7f9d4e6ce453a5039d2ca14e36a02705ed03dd9c7aded43dd7415543140476d5863")

    results = client.search(
    collection_name='test02',
    data = [query_embedding],
    limit=10,
    output_fields=["text"])

    #print(results)
    documents = []

    for result in results[0]:
        documents.append(result['entity']['text'])
    return documents

def getAnswer(query, relevant_documents):
    #get answer from RAG

    openai = OpenAI(api_key="OPENAI API KEY HERE")
    
    PROMPT = f"""You are a helpful assistant. You are assisting a customer with a question.
    You will be given 2 things:
    1. A question
    2. Extracts of text from documents relevant to the question

    PROVIDE AN ANSWER TO THE QUESTION BASED ONLY ON THE EXTRACTS OF TEXT.

    Question:
    {query}

    Relevant Documents:
    {relevant_documents}

"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query}],)

    response = response.choices[0].message.content
    return response

def getAnswerWithRag(query: str):

    query = query.lower()

    #Get embedding of query
    query_embedding = getEmbedding(query)

    #get relevant documents from RAG
    relevant_documents = getRelevantDocuments(query_embedding)
    #print(relevant_documents)

    #get answer from RAG
    answer = getAnswer(query, relevant_documents)

    return answer 

if __name__ == '__main__':
    #get answer from RAG
    query = "What is northwestern University's policy on Gifts?"
    answer = getAnswerWithRag(query)
    print(answer)