import os
import PyPDF2


from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer

from pymilvus import MilvusClient

import hydra
from omegaconf import DictConfig, OmegaConf

def read_data(folder_path):
    pdf_contents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = ""
                for page in pdf_reader.pages:
                    content += "\n \n" + page.extract_text()
                pdf_contents.append(content)
    
    return pdf_contents


#Create a function to convert the list of strings into [[chunk1, chunk2, chunk3, ...], [chunk1, chunk2, chunk3, ...], ...]
def convert_to_chunks(data):
    character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=256,
    chunk_overlap=32
)
    character_split_texts = character_splitter.split_text('\n\n'.join(data))

    
    print(f"\nTotal chunks: {len(character_split_texts)}")

    return character_split_texts

#Create a function to process the [[chunk1, chunk2, chunk3, ...], [chunk1, chunk2, chunk3, ...], ...] by removing stop words, punctuation, etc. from the chunks
def process_chunks(data):
    return data

#Create a function to convert the [[chunk1, chunk2, chunk3, ...], [chunk1, chunk2, chunk3, ...], ...] into list of [[embedding1, embedding2, embedding3, ...], [embedding1, embedding2, embedding3, ...], ...]
def convert_to_embeddings(data):
    model = SentenceTransformer('all-mpnet-base-v2')

    embeddings = []
    for chunk in data:
        embeddings.append(model.encode(chunk))
    return embeddings

def get_milvus_client(endpoint, key):
    client = MilvusClient(uri=endpoint, token=key)

    # # Create a collection
    # client.create_collection(
    #     collection_name=cfg.milvus.collection_name,
    #     dimension=1536
    # )

    return client

#Create a function to store the list of [[embedding1, embedding2, embedding3, ...], [embedding1, embedding2, embedding3, ...], ...] into a VDB
def store_embeddings(collection_name, client, chunks,embeddings):
    data = []
    #Prepare data for insertion
    for (chunk, embedding) in zip(chunks, embeddings):
        data.append({"text": chunk, "Vectorfield": embedding})
    return client.insert(collection_name=collection_name, data=data)

@hydra.main(version_base=None, config_path="../config", config_name="milvus")
def main(cfg):
    data = read_data("Data/")
    chunks = convert_to_chunks(data)
    embeddings = convert_to_embeddings(chunks)
    client = get_milvus_client(cfg.milvus.endpoint, cfg.milvus.key)
    store_embeddings(cfg.milvus.collection_name, client, chunks, embeddings)

if __name__ == '__main__':
    main()

