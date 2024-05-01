import os 
from openai import OpenAI
import pandas as pd
import numpy as np
from query_classify import predict_query_label

domain_description = {
    "NUPolicy": "This domain includes textual data realted to Northwestern University Policies, which include polices in the fields of academic, research, financial, human resources, facilities and safety, information technology."
}

def remove_question_numbers(questions):
    #print(questions)
    cleaned_questions_split = questions.split('. ')

    if(len(cleaned_questions_split) > 1):
        return cleaned_questions_split[1]
    else:
        return questions
    
def query_break_down(user_question):
    client = OpenAI(api_key="")
    subquestions = []
   
    domains = "\n\nDomains:\n"
    for domain, description in domain_description.items():
        domains += f"- {domain}: {description}\n"
    
    prompt = """Your job is to read the followling domain descriptions and according to these domains, break the question into one or more subquestions regarding what, where, when, who, why, OR how, such that if we answer all subquestions, then we can answer the original question.
        Try to make each subqestion related to one domain; otherwise, include it in the list anyway.
        DO NOT NUMBER THE SUBQUESTIONS. Generate subquestions only without any introduction or conclusion. DO NOT add quotation marks or bullet points for subquestions. Questions should be separated by a newline character. Return this list of subquestions and nothing else. """
    prompt += domains
    prompt += f"\nQuestion to be broken down:\n{user_question}"
    # print(prompt)

    response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],)
    responses = response.choices[0].message.content
    # print(responses)
    
    for q in responses.split("\n"):
        subquestion = remove_question_numbers(q)
        subquestions.append(subquestion)
    
    results = [q for q in subquestions if q]
    return results

def load_data(data_file):
    data = pd.read_csv(data_file)
    questions = data['Question'].tolist()
    labels = data['Label'].tolist()

    return questions, labels

if __name__ == '__main__':
    questions, labels = load_data('query.csv')
    # num_pocily_unrelated = labels.count(0)
    # num_pocily_related = labels.count(1)
    pocily_unrelated = []
    pocily_related = []
    for index, question in enumerate(questions):
        subquestions = query_break_down(question)
        pred = 0
        for subquestion in subquestions:
            pred += predict_query_label(subquestion)
        subq_policy_related = pred/len(subquestions)
        
        if labels[index] == 0:
            pocily_unrelated.append(subq_policy_related)
        else:
            pocily_related.append(subq_policy_related)
        
    avg_unrelated = np.sum(np.array(pocily_unrelated))/len(pocily_unrelated)
    avg_related = np.sum(np.array(pocily_related))/len(pocily_related)
    print(f"NU Policy Unreleated Question: {avg_unrelated} subquestions are policy related")
    print(f"NU Policy releated Question: {avg_related} subquestions are policy related")