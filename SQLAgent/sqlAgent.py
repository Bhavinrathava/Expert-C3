import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


class SQL_Agent():
    def __init__(self, question, metadata):
        self.question = question
        self.metadata = metadata
        self.generated_query = None
        self.prompt = "tables:\n" + self.metadata + "\n" + "query for:" + self.question
        # Initialize the tokenizer from Hugging Face Transformers library
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql')
        self.model = self.model.to(self.device)
        self.model.eval()

        
    def run_inference(self):
        # Tokenize the input prompt
        inputs = self.tokenizer(self.prompt, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=512)
        
        # Decode the output IDs to a string (SQL query in this case)
        generated_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_sql


if __name__ == "__main__":
    question = "List the id of students who never attends courses?"
    metadata = "CREATE TABLE student_course_attendance (student_id VARCHAR); CREATE TABLE students (student_id VARCHAR)"
    agent = SQL_Agent(question, metadata)
    print(agent.run_inference())