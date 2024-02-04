import os
import PyPDF2


from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


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
    chunk_size=1000,
    chunk_overlap=0
)
    character_split_texts = character_splitter.split_text('\n\n'.join(data))

    
    print(f"\nTotal chunks: {len(character_split_texts)}")

    return character_split_texts



    pass

#Create a function to process the [[chunk1, chunk2, chunk3, ...], [chunk1, chunk2, chunk3, ...], ...] by removing stop words, punctuation, etc. from the chunks
def process_chunks(data):
    return data

#Create a function to convert the [[chunk1, chunk2, chunk3, ...], [chunk1, chunk2, chunk3, ...], ...] into list of [[embedding1, embedding2, embedding3, ...], [embedding1, embedding2, embedding3, ...], ...]
def convert_to_embeddings(data):

    pass

#Create a function to store the list of [[embedding1, embedding2, embedding3, ...], [embedding1, embedding2, embedding3, ...], ...] into a VDB
def store_embeddings(data):
    pass


def main():
    data = read_data("Data/")
    chunks = convert_to_chunks(data)
    pass

if __name__ == '__main__':
    main()

