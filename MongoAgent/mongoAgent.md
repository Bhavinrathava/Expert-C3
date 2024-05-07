# Seq2NoSQL Agent for Mongo DB

## What does this module do ?
Allows Natural language querying for Mongo Cluster. You can hit the hosted API with get_data/ endpoint with a post request in the requisite format and you will get the data returned from the collection as response. 

## How to use this module ? 
To use this module, you have to : 
- Create folder Config/ in this(MongoAgent) directory
- Add a .pem file for your collection for authentication 
- Create a mongo.env file with following properties :
```
MONGO_URI=mongo_connection_string_here
MONGO_CONFIG_PATH=path_to_pem_here
MONGO_DB=mongo_db_name_here
OLLAMA_BASE_URL=url_for_llama_here
```
- Build docker image with ``` docker build -t mongoapi .```
- Run docker image on appropriate port 
- Hit the API according to the requirement

## API documentation 

**Base URL** : http://[hostname]:8000/

Replace [hostname] with the actual host where the API is deployed.

- **Endpoints**
1. **Create Agent**

    **URL**: /create_agent/
    **Method**: POST
    - **Description**: Initializes a MongoDB agent with the specified configuration.
    - **Request Body**: Requires a JSON object with the following properties:
        - **keys**: Array of strings representing the keys to query.
        - **collectionName**: String representing the MongoDB collection name.
        - **modelName**: String representing the model used for queries. ("T5")
        - **userQuery**: String representing the user's query.
        - **mongoQuestion**: String representing the MongoDB query syntax.
        - **mongodb**: String representing the database name.

2. **Get Answer**

    **URL**: /get_answer/
    **Method**: POST
    **Description**: Retrieves answers based on the user's query and configuration.
    **Request Body**: Same as the Create Agent endpoint.


3. **Get Data**

    **URL**: /get_data/
    **Method**: POST
    **Description**: Retrieves data from MongoDB based on the specified parameters and queries.
    Request Body: Same as the Create Agent endpoint.