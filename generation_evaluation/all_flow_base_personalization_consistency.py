import subprocess
from subprocess import TimeoutExpired

from datasets import load_dataset
import pandas as pd
import os
import random
from tqdm import tqdm
import shutil
import json
import numpy as np
from langchain.schema import HumanMessage

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import multiprocessing

random.seed(42)

# Parameters
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", type=int, default=1)
parser.add_argument("--group", type=str, default="valid")
parser.add_argument("--mode", type=str, default="explain")
parser.add_argument("--evaluation_k", type=int, default=1)
parser.add_argument("--user_id", type=int, default=121793)
args = parser.parse_args()
sample_num = args.sample_num
group = args.group
mode = args.mode
evaluation_k = args.evaluation_k
user_id = args.user_id

code_temperature = 0.2
code_top_p = 0.1
text_temperature = 0.7
text_top_p = 0.8
model = "gpt-3.5-turbo"
# model
# Please set your own OpenAI API key
code_model = ChatOpenAI(model=model, 
    openai_api_base="https://admin.openai.one/v1", 
    api_key="",  
    temperature=code_temperature,
    model_kwargs={"top_p": code_top_p})
text_model = ChatOpenAI(model=model,
    openai_api_base="https://admin.openai.one/v1", 
    api_key="", 
    temperature=text_temperature,
    model_kwargs={"top_p": text_top_p})

# Multi-processing

def run_script_with_input(script_path, input_str, timeout=3):
    try:
        process = subprocess.Popen(["python", script_path], 
                                   stdin=subprocess.PIPE, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   text=True)
        
        stdout, stderr = process.communicate(input=input_str, timeout=timeout)
        
        return stdout, str(stderr)
    except TimeoutExpired:
        process.kill()
        return None, "Execution Timeout. Too slow."
    except Exception as e:
        return None, str(stderr)

def calc_distance_between_results(test_outputs, results):
    d_tot = 0
    for i in range(len(test_outputs)):
        expected = test_outputs[i].rstrip().split("\n")
        if results[i] is None:
            d_tot += len(expected)
        else:
            actual = results[i].rstrip().split("\n")
            if len(actual) != len(expected):
                d_tot += len(expected)
            else:
                t1 = np.array(actual)
                t2 = np.array(expected)
                d_tot += np.sum(t1 != t2)
    return d_tot

def compare(output, real_output):
    if output == real_output:
        return True
    return False

def handle(idx, folder_path, error_index, error_name, user_id):
    df_sample = df.loc[idx, :]
    problem = df_sample["description"]
    name = df_sample["name"]
    input_list = df_sample["public_tests"]["input"]
    output_list = df_sample["public_tests"]["output"]

    solutions = (soln for lang, soln in zip(df_sample["solutions"]["language"], df_sample["solutions"]["solution"]) if lang in (1, 3))
    code = min(solutions, key=len)

    shutil.rmtree(folder_path+name,ignore_errors=True)
    os.makedirs(folder_path+name, exist_ok=True)

    with open(folder_path+name+"/ground_truth.py", "w") as file:
        file.write(code)
    try:
        user_inquiry_history = ""

        with open('user_profile/user_{}_questions.txt'.format(str(user_id)), 'r') as file:
            for line in file:
                user_inquiry_history += line
        profile_analysis_prompt = """Given the user's Stack Overflow question history provided below, analyze and infer the user's programming skills and background.
<user_inquiry_history>
{user_inquiry_history}
</user_inquiry_history>

Consider the following aspects in your analysis:
- Programming Languages: Identify the programming languages the user is familiar with based on the tags and content of their questions.
- Skill Level: Estimate the user's proficiency in these languages. Are their questions beginner, intermediate, or advanced level?
- Topics of Interest: Determine the primary topics or areas the user is interested in (e.g., web development, data science, machine learning, system programming).
- Problem-Solving Approach: Assess the user's problem-solving approach based on the nature and complexity of their questions. Do they demonstrate an understanding of fundamental concepts, or are they focused on specific issues?
- Experience: Gauge the user's overall experience level. Do their questions suggest they are a student, hobbyist, junior developer, or senior developer?
- Other Relevant Information: Note any other relevant information that can help in understanding the user's background (e.g., specific technologies or frameworks they frequently use).

Please response in the following JSON format:
{{
    "programming_languages": "",
    "skill_level": "",
    "topics_of_interest": "",
    "problem_solving_approach": "",
    "experience": "",
    "other_relevant_information": ""
}}
"""
        response = text_model([HumanMessage(content=profile_analysis_prompt.format(user_inquiry_history=user_inquiry_history))])
        profile_analysis = json.loads(response.content)

        with open(folder_path+name+"/profile_analysis_prompt.txt", "w") as file:
            file.write(profile_analysis_prompt.format(user_inquiry_history=user_inquiry_history))
        with open(folder_path+name+"/profile_analysis.txt", "w") as file:
            file.write(json.dumps(profile_analysis))


        reflection_prompt = """You are given a code contest problem:
<problem_name>{name}</problem_name>
<problem_description>
{problem}
</problem_description>

Given the code contest problem, you should reflect on the problem, and describe it in your own words. Pay attention to small details, nuances, notes and examples in the problem description.

{format_instructions}"""

        response_schemas = [
            ResponseSchema(name="self_reflection", description="Describe the problem in your own words. Address the problem goals, inputs, outputs, rules, constraints, and other relevant details."),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template=reflection_prompt,
            input_variables=["name", "problem"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | text_model | parser
        reflection = chain.invoke({"name": name, "problem": problem})

        reflection_dict = {
            "name": name,
            "problem": problem,
            "format_instructions": format_instructions,
        }
        with open(folder_path+name+"/self_reflection_prompt.txt", "w") as file:
            file.write(reflection_prompt.format(**reflection_dict))
        with open(folder_path+name+"/self_reflection.txt", "w") as file:
            file.write(json.dumps(reflection))

        explanation_prompt = """Your task is to comprehend a competitive programming problem and interpret its solution, trying to meet the given user's individual programming skills and background. 
<profile_analysis>{profile_analysis}</profile_analysis>

You are given a code contest problem, and a self-reflection on the problem:
<problem_name>{name}</problem_name>
<problem_description>{problem}</problem_description>
<self_reflection>{self_reflection}</self_reflection>

Additionally, you are given an accepted correct solution:
<code>{code}</code>

You are also given an input list of test cases and an expected output list:
<input_list>{input_list}</input_list>
<expected_output_list>{output_list}</expected_output_list>

Guidelines:
- Review and consider the given information above.
- Aim for clarity in your description.
- Each section should be self-explanatory, meaning each point should be clear even without reference to other points.

Let's think step-by-step.
- First, provide a step-by-step description of the solution.
- Next, give a high-level explanation of the solution.
- Finally, provide a personalized explanation of the solution.

For the step-by-step description, consider the following requirements:
- Describe the purpose of each variable used in the code. Clarify the initial state and value assignments.
- Break down the logic in the code step-by-step. Explain the purpose and effect of each significant block of code. Show how these steps collectively lead to the final solution.

For the high-level explanation, consider the following requirements:
- Provide a concise overview of the solution's purpose and functionality. 
- Highlight the main components or algorithms used in the solution.
- Describe how this algorithm addresses the problem statement effectively.

For the personalized explanation, consider the following requirements:
- Use language and examples relevant to the user's skill level and background. Simplify complex concepts where necessary, relating them to the user's known topics of interest.
- Explain how understanding this solution can benefit the user. Relate the solution to real-world applications or similar problems the user might encounter.

{format_instructions}"""

        response_schemas = [
            ResponseSchema(name="step_by_step_description", description="Provide a step-by-step description of the solution."),
            ResponseSchema(name="high_level_explanation", description="Provide a high-level explanation of the solution."),
            ResponseSchema(name="personalized_explanation", description="Provide a personalized explanation of the solution."),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template=explanation_prompt,
            input_variables=["profile_analysis", "name", "problem", "self_reflection", "input_list", "output_list", "code"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | text_model | parser

        explanation_dict = {
            "profile_analysis": json.dumps(profile_analysis),
            "name": name,
            "problem": problem,
            "input_list": input_list,
            "self_reflection": reflection["self_reflection"], 
            "output_list": output_list,
            "code": code,
            "format_instructions": format_instructions,
        }
        with open(folder_path+name+"/explanation_prompt.txt", "w") as file:
            file.write(explanation_prompt.format(**explanation_dict))

        explanation_list = []
        for i in range(4):

            explanation = chain.invoke({
                "profile_analysis": json.dumps(profile_analysis),
                "name": name, 
                "problem": problem, 
                "self_reflection": reflection["self_reflection"], 
                "input_list": input_list, 
                "output_list": output_list,
                "code": code
            })


            explanation_list.append(json.dumps(explanation))
            with open(folder_path+name+"/explanation_{}.txt".format(i), "w") as file:
                file.write(json.dumps(explanation))

        decide_prompt = """Imagine you are a user with the following programming skills and background:
<profile_analysis>{profile_analysis}</profile_analysis>
        
You are given a code contest problem, a self-reflection on the problem, an accepted correct solution, and five explanations of the solution:
<problem_name>{name}</problem_name>
<problem_description>{problem}</problem_description>
<self_reflection>{self_reflection}</self_reflection>
<code>{code}</code>
[
    {{
        "method": "method_0",
        "explanation": {explanation_0}
    }},
    {{
        "method": "method_1",
        "explanation": {explanation_1}
    }},
    {{
        "method": "method_2",
        "explanation": {explanation_2}
    }},
    {{
        "method": "method_3",
        "explanation": {explanation_3}
    }}
]

Your task is to create a leaderboard of the explanations based on the following criteria, so that the best explanation is ranked first:
- Clarity: The explanation is clear and easy to understand.
- Completeness: The explanation covers all the necessary details.
- Accuracy: The explanation accurately describes the solution.
- Relevance: The explanation is relevant to the problem and the solution.
- Alignment: The explanation aligns with your programming skills and background.

Please provide your decision and why in the following JSON format:
{{
    "leaderboard": [], // A sorted list of the methods from best to worst, e.g., ["method_0", "method_1", "method_2", "method_3"]
    "reasoning": ""
}}"""

        response = text_model([HumanMessage(content=decide_prompt.format(profile_analysis=json.dumps(profile_analysis), name=name, problem=problem, self_reflection=reflection["self_reflection"], code=code, explanation_0=explanation_list[0], explanation_1=explanation_list[1], explanation_2=explanation_list[2], explanation_3=explanation_list[3]))])
        decision = json.loads(response.content)
        
        with open(folder_path+name+"/decision.txt", "w") as file:
            file.write(json.dumps(decision))

        index = int(decision["leaderboard"][0].split("_")[1])
        best_explanation = json.loads(explanation_list[index])
        
        with open(folder_path+name+"/best_explanation.txt", "w") as file:
            file.write(json.dumps(best_explanation))
    
    except Exception as e:
        print(e)
        error_index.append(idx)
        error_name.append(name)

def evaluation(idx, evaluation_k, evaluation_folder_path, folder_path, error_index, error_name):
    df_sample = df.loc[idx, :]
    problem = df_sample["description"]
    name = df_sample["name"]
    input_list = df_sample["private_tests"]["input"] if df_sample["private_tests"]["input"] else df_sample["generated_tests"]["input"]
    output_list = df_sample["private_tests"]["output"] if df_sample["private_tests"]["output"]  else df_sample["generated_tests"]["output"]
    
    shutil.rmtree(evaluation_folder_path+name,ignore_errors=True)
    os.makedirs(evaluation_folder_path+name, exist_ok=True)

    with open(folder_path+name+"/best_explanation.txt", "r") as file:
        explanation_txt = file.read()
        explanation = json.loads(explanation_txt)
        keys_to_extract = ["step_by_step_description", "high_level_explanation"]
        explanation = {key: explanation[key] for key in keys_to_extract}
        
    evaluation_prompt = """You are given a code contest problem:
<problem_name>{name}</problem_name>
<problem_description>
{problem}
</problem_description>

The following is a hint that can lead to one correct solution of the problem:
<hint>
{explanation}
</hint>

Your task is to read and understand the problem, analyze the hint and how to use it to solve the problem, think of a solution accordingly and complete the python code of the solution.

Code requirements:
- Make sure to include all the necessary module imports, properly initialize the variables, and address the problem constraints.
- The code needs to be self-contained, and executable as-is.
- The code should read the input using the "input()" method. Make sure to properly parse the input, according to the problem description.
- The output should be printed without additional words using the "print()" method.

The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```python" and "```":

```python
def f1(...):
    ...
    return ...

def f2(...):
    ...
    return ...

...

if __name__ == "__main__":
    ...
```"""

    eval_dict = {
        "name": name,
        "problem": problem,
        "explanation": json.dumps(explanation),
    }

    with open(evaluation_folder_path+name+"/evaluation_prompt.txt", "w") as file:
        file.write(evaluation_prompt.format(**eval_dict))

    try:
    
        success_count = 0
        for i in range(evaluation_k):

            prompt = PromptTemplate(
                template=evaluation_prompt,
                input_variables=["name", "problem", "explanation"],
            )

            chain = prompt | code_model
            generated_code = chain.invoke({"name": name, "problem": problem, "explanation": json.dumps(explanation)}).content


            # clean up the response
            generated_code = generated_code[10:]
            generated_code = generated_code.rstrip("\n```")

            with open(evaluation_folder_path+name+"/evaluation_{}.py".format(i), "w") as file:
                file.write(generated_code)
            
            script_path = evaluation_folder_path+name+"/evaluation_{}.py".format(i)
            real_output_list = []
            for input_str in input_list:
                if run_script_with_input(script_path, input_str)[1] == "Execution Timeout. Too slow.":
                    current_len = len(real_output_list)
                    break
                real_output_list.append(run_script_with_input(script_path, input_str)[0])
            if len(real_output_list) < len(output_list):
                real_output_list = real_output_list + [None] * (len(output_list) - current_len)
            
            if compare(output_list, real_output_list):
                success_count = 1
                break
        return success_count
    except Exception as e:
        print(e)
        error_index.append(idx)
        error_name.append(name)
        return None

def print_error(value):
    print("Multiprocessing Error: ", value)

if __name__ == "__main__":
    # Dataset
    data = load_dataset("deepmind/code_contests", split=group)
    df = pd.DataFrame(data)
    target_idx_list = [idx for idx, value in df.iterrows() if "<image>" not in value["description"] and (3 in value["solutions"]["language"] or 1 in value["solutions"]["language"])]

    folder_path = "./result/explanation/{}/base_personalization_consistency/{}/sample=".format(group, str(user_id)) + str(sample_num) + "/"
    os.makedirs(folder_path, exist_ok=True)

    if mode == "explain":
        # Generation Global Variables
        manager = multiprocessing.Manager()
        error_index = manager.list()
        error_name = manager.list()

        pbar = tqdm(total=len(target_idx_list))
        pbar.set_description("Processing Sample: {}".format(sample_num))
        update = lambda *args: pbar.update()

        pool = multiprocessing.Pool(processes=10)
        [pool.apply_async(handle, args=(idx, folder_path, error_index, error_name, user_id,), callback=update, error_callback=print_error) for idx in target_idx_list]
        pool.close()
        pool.join()

        print("Error Name: ", error_name)
        print("Error Index: ", error_index)

        while error_index:
            temp_error_index = list(error_index)
            
            pbar = tqdm(total=len(temp_error_index))
            pbar.set_description("ErrorProcessing Sample: {}".format(sample_num))
            update = lambda *args: pbar.update()

            
            error_index[:] = []
            error_name[:] = []
            pool = multiprocessing.Pool(processes=10)
            [pool.apply_async(handle, args=(idx, folder_path, error_index, error_name, user_id,), callback=update, error_callback=print_error) for idx in temp_error_index]
            pool.close()
            pool.join()
            print("Error Name: ", error_name)
            print("Error Index: ", error_index)

    
    if mode == "evaluate":
        # Evaluation Global Variables
        evaluation_folder_path = "./result/evaluation/{}/k=".format(group) + str(evaluation_k) + "/base_personalization_consistency/{}/sample=".format(str(user_id)) + str(sample_num) + "/"
        os.makedirs(evaluation_folder_path, exist_ok=True)

        success_count = 0
        manager = multiprocessing.Manager()
        error_index = manager.list()
        error_name = manager.list()

        pbar = tqdm(total=len(target_idx_list))
        pbar.set_description("Evaluation @ {} Sample: {}".format(evaluation_k, sample_num))
        update = lambda *args: pbar.update()

        pool = multiprocessing.Pool(processes=10)
        evaluation_results = [pool.apply_async(evaluation, args=(idx, evaluation_k, evaluation_folder_path, folder_path, error_index, error_name,), callback=update, error_callback=print_error) for idx in target_idx_list] 
        pool.close()
        pool.join()

        all_count = 0
        for item in evaluation_results:
            if item.get() is not None:
                success_count += item.get()
                all_count += 1

        print("Success Evaluation Count: ", all_count)
        print("Error Name: ", error_name)
        print("Error Index: ", error_index)

        retry = 0
        while error_index:

            temp_error_index = list(error_index)
            
            pbar = tqdm(total=len(temp_error_index))
            pbar.set_description("ReEvaluation @ {} Sample: {}".format(evaluation_k, sample_num))
            update = lambda *args: pbar.update()

            error_index[:] = []
            error_name[:] = []
            pool = multiprocessing.Pool(processes=10)
            evaluation_results = [pool.apply_async(evaluation, args=(idx, evaluation_k, evaluation_folder_path, folder_path, error_index, error_name,), callback=update, error_callback=print_error) for idx in temp_error_index] 
            pool.close()
            pool.join()
            print("Error Name: ", error_name)
            print("Error Index: ", error_index)
            for item in evaluation_results:
                if item.get():
                    success_count += item.get()

        print("Success N @ {}: ".format(evaluation_k), success_count)
        print("Total N @ {}: ".format(evaluation_k), len(target_idx_list))
        print("Success Rate @ {}: ".format(evaluation_k), success_count * 1.0/len(target_idx_list))
      