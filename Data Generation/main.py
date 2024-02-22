import langchain
import PyPDF2
import os 
from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
def extract_text_from_pdf(pdf_path):

    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        contents = ""
        for page in pdf_reader.pages:
            contents += "\n \n" + page.extract_text()
        
    return contents

def remove_question_numbers(questions):
    #print(questions)
    cleaned_questions_split = questions.split('. ')

    if(len(cleaned_questions_split) > 1):
        return cleaned_questions_split[1]
    else:
        return questions

def main():
    client = OpenAI()
    folder_path = 'Data/'
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    

    prompt ="""Your job is to read the following text extract that has been prepared from a PDF file. 
    Based on the text, you have to generate 5 questions. DO NOT NUMBER THE QUESTIONS. Generate 5 questions only without any 
    introduction or conclusion. Questions should be separated by a newline character. Return this list of 5 questions and nothing else. 
    PDF Extract:


    """

    character_splitter = RecursiveCharacterTextSplitter(
separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=30)

    
    

    for fileName in files:
        
        print("Processing the file: ", fileName)
        pdf_contents = extract_text_from_pdf(folder_path + fileName)

        #split the content into chunks 
        chunks = character_splitter.split_text(pdf_contents)
        count = 1
        for pdf_content in chunks:
            try:
                masterquestions = []
                current_prompt = prompt + pdf_content
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": current_prompt}],)
                
                responses = response.choices[0].message.content

                for response in responses.split("\n"):
                    response = remove_question_numbers(response)
                    masterquestions.append(response)

                #Write the questions to a csv file
                with open("generated_questions"+fileName+str(count)+".csv", "w") as file:
                    for question in masterquestions:
                        file.write(question + "\n")
            except Exception as e:
                print(e)

            count += 1

        #Making the final prompt : 
        prompt += pdf_contents



        #print(masterquestions)

if __name__ == '__main__':
    main()




