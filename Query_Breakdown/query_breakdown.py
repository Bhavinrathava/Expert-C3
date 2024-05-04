import os 
from openai import OpenAI
import pandas as pd
import numpy as np
# from NUPolicy_query_classify import predict_query_label
import json
import regex

domain_description = {
    "NUPolicy": "This domain includes textual data related to Northwestern University Policies, which include polices in the fields of academic, research, financial, human resources, facilities and safety, information technology.",
    "Events": "This domain includes data related to the events and activities scheduled at Northwestern University campus, including their theme, category, time, location, hosting department, audience and description.",
}

# Return a list [{"Query":subquestion, "Domain": domain_name}, ...]
def query_breakdown(user_question):
    client = OpenAI(api_key="")
   
    domains = "\n\nDomains:\n"
    for domain, description in domain_description.items():
        domains += f"- {domain}: {description}\n"
    
    prompt = """Your job is to read the followling domain descriptions and according to these domains, break the question into one or more subquestions regarding what, where, when, who, why, OR how, such that if we answer all subquestions, then we can answer the original question.
        Try to make each subqestion related to one domain; if the subquestion is unrelated to any domain, do not include it.
        You can have multiple subqusetsions have the same domain.
        Please return the result in a json, with the domain name as the key, and the corresponding subquestion as the value, see the format below.
        In the json, do not empty key-value pair, do not include any "subquestion" wording.
        Do not include any introduction or conclusion. Just return the json.
        Format to return:
        {"Domain_name_1": "Subquestion_1", "Domain_name_2": "Subquestion_2", "Domain_name_2": "Subquestion_2", ...}
        """
    prompt += domains
    prompt += f"\nQuestion to be broken down:\n{user_question}"

    success = False
    while not success:
        response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],)
        responses = response.choices[0].message.content
        
        pattern = r"\{(?:[^{}]|(?R))*\}"
        breakdown_str = regex.search(pattern, responses).group()
        if is_valid_json(breakdown_str):
            results = process_json(breakdown_str)
            success = True

    return results


def remove_question_numbers(questions):
    cleaned_questions_split = questions.split('. ')

    if(len(cleaned_questions_split) > 1):
        return cleaned_questions_split[1]
    else:
        return questions

def process_json(breakdown_str):
    breakdown_dict = json.loads(breakdown_str)
    subq_domain = []
    for domain, subquestion in breakdown_dict.items():
        if domain in domain_description.keys() and subquestion != '' and subquestion is not None:
            pair = {"Query": subquestion, "Domain": domain}
            subq_domain.append(pair)

    return subq_domain


# Check whether a str can be loaded to json format
def is_valid_json(breakdown_str):
    try:
        json.loads(breakdown_str)
        return True
    except json.JSONDecodeError:
        return False

def load_data(data_file):
    data = pd.read_csv(data_file)
    questions = data['Question'].tolist()
    labels = data['Label'].tolist()

    return questions, labels

if __name__ == '__main__':
    question = "I want to host a sports competition, what should I do?"
    subq = query_breakdown(question)
    print(subq)
