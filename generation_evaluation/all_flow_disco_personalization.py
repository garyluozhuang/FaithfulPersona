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
import time
parser = argparse.ArgumentParser()
parser.add_argument("--sample_num", type=int, default=1)
parser.add_argument("--iter_count", type=int, default=1)
parser.add_argument("--group", type=str, default="valid")
parser.add_argument("--mode", type=str, default="explain")
parser.add_argument("--evaluation_k", type=int, default=1)
parser.add_argument("--user_id", type=int, default=121793)
args = parser.parse_args()
sample_num = args.sample_num
iter_count = args.iter_count
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
# Please set your OpenAI API key here
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

def handle(idx, iter_count, folder_path, error_index, error_name, user_id):
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

        explanation_prompt = """Your task is to comprehend a competitive programming problem and interpret its solution.

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

For the step-by-step description, consider the following requirements:
- Describe the purpose of each variable used in the code. Clarify the initial state and value assignments.
- Break down the logic in the code step-by-step. Explain the purpose and effect of each significant block of code. Show how these steps collectively lead to the final solution.

For the high-level explanation, consider the following requirements:
- Provide a concise overview of the solution's purpose and functionality. 
- Highlight the main components or algorithms used in the solution.
- Describe how this algorithm addresses the problem statement effectively.

{format_instructions}"""

        response_schemas = [
            ResponseSchema(name="step_by_step_description", description="Provide a step-by-step description of the solution."),
            ResponseSchema(name="high_level_explanation", description="Provide a high-level explanation of the solution."),
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        prompt = PromptTemplate(
            template=explanation_prompt,
            input_variables=["name", "problem", "self_reflection", "input_list", "output_list", "code"],
            partial_variables={"format_instructions": format_instructions},
        )
        chain = prompt | text_model | parser
        explanation = chain.invoke({
            "name": name, 
            "problem": problem, 
            "self_reflection": reflection["self_reflection"], 
            "input_list": input_list, 
            "output_list": output_list,
            "code": code
        })

        explanation_dict = {
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
        with open(folder_path+name+"/explanation.txt", "w") as file:
            file.write(json.dumps(explanation))

        validation_prompt = """You are given a code contest problem:
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

        prompt = PromptTemplate(
            template=validation_prompt,
            input_variables=["name", "problem", "explanation"],
        )

        chain = prompt | code_model
        generated_code = chain.invoke({"name": name, "problem": problem, "explanation": json.dumps(explanation)}).content

        # clean up the response
        generated_code = generated_code[10:]
        generated_code = generated_code.rstrip("\n```")

        validation_dict = {
            "name": name,
            "problem": problem,
            "explanation": json.dumps(explanation),
        }
            
        with open(folder_path+name+"/initial_validation_prompt.txt", "w") as file:
            file.write(validation_prompt.format(**validation_dict))
        with open(folder_path+name+"/initial_validation.py", "w") as file:
            file.write(generated_code)


        script_path = folder_path+name+"/initial_validation.py"
        real_output_list = []
        err_flag = False
        for input_str in input_list:
            real_output, err = run_script_with_input(script_path, input_str)
            real_output_list.append(real_output)
            if err != "":
                err_flag = True
                run_into_error = err
        distance = calc_distance_between_results(output_list, real_output_list)
        calc_distance = calc_distance_between_results(output_list, real_output_list)
        with open(folder_path+name+"/distance.txt", "w") as file:
            file.write(f"{distance}\t{calc_distance}\n")

        best_distance = distance
        best_explanation = explanation
    
        failure_analysis_prompt = """You are given a code contest problem and a self-reflection on the problem:
<problem_name>{name}</problem_name>
<problem_description>
{problem}
</problem_description>

<self_reflection>
{self_reflection}
</self_reflection>

Additionnally, you are given an accepted correct solution:
<code>
{code}
</code>

You are also given an input list of test cases and an expected output list:
<input_list>
{input_list}
</input_list>

<expected_output_list>
{output_list}
</expected_output_list>

A Python code solution was generated for the problem:
<implemented_code>
{implemented_code}
</implemented_code>

However, when running on the given input, the code solution above failed to produce the expected output:
<incorrect_output_list>
{incorrect_output_list}
</incorrect_output_list>

Your task is to analyze the failure.

Let's think step by step:
- First, find the differences between the incorrect output list and the expected output list, and explain why the coder's solution caused these discrepancies.
- Then, find out the problem with the implemented code compared to the accepted correct solution.

{format_instructions}"""

        err_failure_analysis_prompt = """You are given a code contest problem and a self-reflection on the problem:
<problem_name>{name}</problem_name>
<problem_description>
{problem}
</problem_description>

<self_reflection>
{self_reflection}
</self_reflection>

Additionnally, you are given an accepted correct solution:
<code>
{code}
</code>

You are also given an input list of test cases and an expected output list:
<input_list>
{input_list}
</input_list>

<expected_output_list>
{output_list}
</expected_output_list>

A Python code solution was generated for the problem:
<implemented_code>
{implemented_code}
</implemented_code>

However, when running on the given input, the code solution above failed to produce the expected output and raised an error during the execution:
<incorrect_output_list>
{incorrect_output_list}
</incorrect_output_list>

<error>
{run_into_error}
</error>

Your task is to analyze the failure.

Let's think step by step:
- First, find the differences between the incorrect output list and the expected output list, analyze the provided error list, and explain why the coder's solution caused these discrepancies and errors.
- Then, find out the problem with the implemented code compared to the accepted correct solution.

{format_instructions}"""

        fix_prompt = """You are given a code contest problem, a self-reflection on the problem, an input list of test cases and an expected output list:
<problem_name>{name}</problem_name>
<problem_description>{problem}</problem_description>
<self_reflection>{self_reflection}</self_reflection>
<input_list>{input_list}</input_list>
<expected_output_list>{output_list}</expected_output_list>

You are also given one correct solution to the problem and its explanation:
<code>{code}</code>
<explanation>{explanation}</explanation>

Your task is to revise the provided explanation based on one kind of feedback.

A coder attempted to implement a solution with the hint of the explanation above. However, the code did not yield the correct output:
<implemented_code>{implemented_code}</implemented_code>
<incorrect_output_list>{incorrect_output_list}</incorrect_output_list>
Here's also a failure analysis of the coder's solution:
<failure_analysis>{failure_analysis}</failure_analysis>

Please revise the provided explanation, considering the failure analysis of the coder's solution.
When other coders read your revised explanation, they should avoid the same mistakes made by the coder who failed to produce the correct solution. 

Guidelines:
- Review and consider the given information above.
- Aim for clarity in your description.
- Each section should be self-explanatory, meaning each point should be clear even without reference to other points.

{format_instructions}"""
    
        err_fix_prompt = """You are given a code contest problem, a self-reflection on the problem, an input list of test cases and an expected output list:
<problem_name>{name}</problem_name>
<problem_description>{problem}</problem_description>
<self_reflection>{self_reflection}</self_reflection>
<input_list>{input_list}</input_list>
<expected_output_list>{output_list}</expected_output_list>

You are also given one correct solution to the problem and its explanation:
<code>{code}</code>
<explanation>{explanation}</explanation>

Your task is to revise the provided explanation based on two kinds of feedbacks.

A coder attempted to implement a solution with the hint of the explanation above. However, the code did not yield the correct output and raised an error during the execution:
<implemented_code>{implemented_code}</implemented_code>
<incorrect_output_list>{incorrect_output_list}</incorrect_output_list>
<error>{run_into_error}</error>
Here's also a failure analysis of the coder's solution:
<failure_analysis>{failure_analysis}</failure_analysis>

Please revise the provided explanation, considering the failure analysis of the coder's solution.
When other coders read your revised explanation, they should avoid the same mistakes made by the coder who failed to produce the correct solution. 


Guidelines:
- Review and consider the given information above.
- Aim for clarity in your description.
- Each section should be self-explanatory, meaning each point should be clear even without reference to other points.

{format_instructions}"""

        n = 0
        while not compare(output_list, real_output_list):
            
            response_schemas = [
                ResponseSchema(name="error_analysis", description="Explain why the coder's solution caused the discrepancies and the errors." if err_flag else "Explain why the coder's solution caused the discrepancies."),
                ResponseSchema(name="what_went_wrong", description="Identify the problem with the implemented code compared to the accepted correct solution."),
            ]
            parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = parser.get_format_instructions()
            if err_flag:
                prompt = PromptTemplate(
                    template=err_failure_analysis_prompt,
                    input_variables=["name", "problem", "self_reflection", "input_list", "output_list", "code", "implemented_code", "incorrect_output_list", "run_into_error"],
                    partial_variables={"format_instructions": format_instructions},
                )
                invoke_dict = {"name": name, "problem": problem, "self_reflection": reflection["self_reflection"], "input_list": input_list, "output_list": output_list, "code": code, "implemented_code": generated_code, "incorrect_output_list": real_output_list, "run_into_error": run_into_error}
            else:
                prompt = PromptTemplate(
                    template=failure_analysis_prompt,
                    input_variables=["name", "problem", "self_reflection", "input_list", "output_list", "code", "implemented_code", "incorrect_output_list"],
                    partial_variables={"format_instructions": format_instructions},
                )
                invoke_dict = {"name": name, "problem": problem, "self_reflection": reflection["self_reflection"], "input_list": input_list, "output_list": output_list, "code": code, "implemented_code": generated_code, "incorrect_output_list": real_output_list}
            chain = prompt | text_model | parser
            failure_analysis = chain.invoke(invoke_dict)

            invoke_dict["format_instructions"] = format_instructions
            with open(folder_path+name+"/failure_analysis_iteration_{}_prompt.txt".format(n+1), "w") as file:
                file.write(failure_analysis_prompt.format(**invoke_dict))
            with open(folder_path+name+"/failure_analysis_iteration_{}.txt".format(n+1), "w") as file:
                file.write(json.dumps(failure_analysis))


            base_input_var = ["name", "problem", "self_reflection" "input_list", "output_list", "code", "explanation", "implemented_code", "incorrect_output_list", "failure_analysis"]

            invoke_dict = {
                "name": name,
                "problem": problem,
                "self_reflection": reflection["self_reflection"],
                "input_list": input_list,
                "output_list": output_list,
                "code": code,
                "explanation": json.dumps(best_explanation),
                "implemented_code": generated_code,
                "incorrect_output_list": real_output_list,
                "failure_analysis": json.dumps(failure_analysis),
            }
                
            if err_flag:
                prompt_template = err_fix_prompt
                input_variables = base_input_var +  ["run_into_error"]
                invoke_dict["run_into_error"] = run_into_error
            else:
                prompt_template = fix_prompt
                input_variables = base_input_var
            
            response_schemas = [
                ResponseSchema(name="step_by_step_description", description="Revised step-by-step description of the solution."),
                ResponseSchema(name="high_level_explanation", description="Revised high-level explanation of the solution."),
            ]
            parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = parser.get_format_instructions()
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=input_variables,
                partial_variables={"format_instructions": format_instructions},
            )
            chain = prompt | text_model | parser
            explanation = chain.invoke(invoke_dict)

            invoke_dict["format_instructions"] = format_instructions
            with open(folder_path+name+"/code_explanation_revision_iteration_{}_prompt.txt".format(n+1), "w") as file:
                file.write(prompt_template.format(**invoke_dict))
            with open(folder_path+name+"/code_explanation_revision_iteration_{}.txt".format(n+1), "w") as file:
                file.write(json.dumps(explanation))
            prompt = PromptTemplate(
                template=validation_prompt,
                input_variables=["name", "problem", "explanation"],
            )

            chain = prompt | code_model
            generated_code = chain.invoke({"name": name, "problem": problem, "explanation": json.dumps(explanation)}).content

            # clean up the response
            generated_code = generated_code[10:]
            generated_code = generated_code.rstrip("\n```")

            validation_dict = {
                "name": name,
                "problem": problem,
                "explanation": json.dumps(explanation),
            }
            
            with open(folder_path+name+"/validation_iteration_{}_prompt.txt".format(n+1), "w") as file:
                file.write(validation_prompt.format(**validation_dict))
            with open(folder_path+name+"/validation_iteration_{}.py".format(n+1), "w") as file:
                file.write(generated_code)
            
            script_path = folder_path+name+"/validation_iteration_{}.py".format(n+1)
            real_output_list = []
            err_flag = False
            
            for input_str in input_list:
                real_output, err = run_script_with_input(script_path, input_str)
                real_output_list.append(real_output)
                if err != "":
                    err_flag = True
                    run_into_error = err

            distance = calc_distance_between_results(output_list, real_output_list)
            calc_distance = calc_distance_between_results(output_list, real_output_list)
            with open(folder_path+name+"/distance.txt", "a") as file:
                file.write(f"{distance}\t{calc_distance}\n")

            if distance <= best_distance:
                best_distance = distance
                best_explanation = explanation


            n += 1
            if n == iter_count:
                break
        
        with open(folder_path+name+"/best_explanation_temp.txt", "w") as file:
            file.write(json.dumps(best_explanation))

        user_inquiry_history = ""
        with open('user_{}_questions.txt'.format(str(user_id)), 'r') as file:
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
        
        personalization_prompt = """Your task is to personalize the explanation of the solution based on the following user's programming skills and background:
<profile_analysis>{profile_analysis}</profile_analysis>

You are given a code contest problem:
<problem_name>{name}</problem_name>
<problem_description>{problem}</problem_description>

You are also given one correct solution to the problem and its explanation:
<code>{code}</code>
<explanation>{explanation}</explanation>

You should provide a personalized explanation of the solution based on the user's programming skills and background and the explanation provided above.

Consider the following requirements in your personalized explanation:
- Use language and examples relevant to the user's skill level and background. Simplify complex concepts where necessary, relating them to the user's known topics of interest.
- Explain how understanding this solution can benefit the user. Relate the solution to real-world applications or similar problems the user might encounter.

Please response in the following JSON format:
{{
    "personalized_explanation": ""
}}
"""
        response = text_model([HumanMessage(content=personalization_prompt.format(profile_analysis=json.dumps(profile_analysis), name=name, problem=problem, code=code, explanation=json.dumps(best_explanation)))])
        personalized_explanation = json.loads(response.content)

        with open(folder_path+name+"/personalization_prompt.txt", "w") as file:
            file.write(personalization_prompt.format(profile_analysis=json.dumps(profile_analysis), name=name, problem=problem, code=code, explanation=json.dumps(best_explanation)))
        with open(folder_path+name+"/personalized_explanation.txt", "w") as file:
            file.write(json.dumps(personalized_explanation))

        rating_prompt = """Imagine you are a programmer with the following individual programming skills and background:
<profile_analysis>
{profile_analysis}
</profile_analysis>

Your are given a code contest problem:
<problem_name>{name}</problem_name>
<problem_description>
{problem}
</problem_description>

You are also given an accepted correct solution:
<code>
{code}
<\code>

Your task is to rate the following personalized explanation of the solution based on how well it aligns with your programming skills and background:
<personalized_explanation>
{personalized_explanation}
</personalized_explanation>

The score should be in the range of 1 to 5.
if the rating is under 5, please provide some revision suggestions in the reasoning section.
Please provide your rating and reasoning in the following JSON format:
{{
    "rating": "",
    "reasoning": ""
}}
"""
        response = text_model([HumanMessage(content=rating_prompt.format(profile_analysis=profile_analysis, name=name, problem=problem, code=code, personalized_explanation=json.dumps(personalized_explanation)))])
        rating = json.loads(response.content)

        with open(folder_path+name+"/initial_rating_prompt.txt", "w") as file:
            file.write(rating_prompt.format(profile_analysis=profile_analysis, name=name, problem=problem, code=code, personalized_explanation=json.dumps(personalized_explanation)))
        with open(folder_path+name+"/initial_rating.txt", "w") as file:
            file.write(json.dumps(rating))

        rating_diff = str(5 - int(rating["rating"]))
        with open(folder_path+name+"/rating_diff.txt", "w") as file:
            file.write(f"{rating_diff}\n")

        best_rating_diff = 5 - int(rating["rating"])
        best_personalized_explanation = personalized_explanation
        
        personalization_fix_prompt = """Your are given a code contest problem:
<problem_name>{name}</problem_name>
<problem_description>{problem}</problem_description>

You are also given one correct solution to the problem and its explanation:
<code>{code}</code>
<explanation>{explanation}</explanation>

A personalized explanation of the solution was generated for a user with the following programming skills and background:
<profile_analysis>{profile_analysis}<\profile_analysis>

This is the generated personalized explanation:
<personalized_explanation>{personalized_explanation}<\personalized_explanation>

The user rated the personalized explanation and provided some revision suggestions:
<rating>{rating}<\rating>

Please revise the personalized explanation based on the user's feedback so that the user will rate the revised personalized explanation higher.

Please response in the following JSON format:
{{
    "personalized_explanation": ""
}}
"""
        n = 0 
        while int(rating["rating"]) < 5:
            response = text_model([HumanMessage(content=personalization_fix_prompt.format(name=name, problem=problem, code=code, explanation=json.dumps(best_explanation), personalized_explanation=json.dumps(personalized_explanation), profile_analysis=json.dumps(profile_analysis), rating=json.dumps(rating)))])
            personalized_explanation = json.loads(response.content)

            with open(folder_path+name+"/personalization_fix_iteration_{}_prompt.txt".format(n+1), "w") as file:
                file.write(personalization_fix_prompt.format(name=name, problem=problem, code=code, explanation=json.dumps(best_explanation), personalized_explanation=json.dumps(personalized_explanation), profile_analysis=json.dumps(profile_analysis), rating=json.dumps(rating)))
            with open(folder_path+name+"/personalization_fix_iteration_{}.txt".format(n+1), "w") as file:
                file.write(json.dumps(personalized_explanation))
            

            response = text_model([HumanMessage(content=rating_prompt.format(profile_analysis=profile_analysis, name=name, problem=problem, code=code, personalized_explanation=json.dumps(personalized_explanation)))])
            rating = json.loads(response.content)
            
            with open(folder_path+name+"/rating_iteration_{}_prompt.txt".format(n+1), "w") as file:
                file.write(rating_prompt.format(profile_analysis=profile_analysis, name=name, problem=problem, code=code, personalized_explanation=json.dumps(personalized_explanation)))
            with open(folder_path+name+"/rating_iteration_{}.txt".format(n+1), "w") as file:
                file.write(json.dumps(rating))

            rating_diff = str(5 - int(rating["rating"]))
            with open(folder_path+name+"/rating_diff.txt", "a") as file:
                file.write(f"{rating_diff}\n")
            if 5 - int(rating["rating"]) <= best_rating_diff:
                best_rating_diff = 5 - int(rating["rating"])
                best_personalized_explanation = personalized_explanation
            
            n+=1
            if n == iter_count:
                break
        
        with open(folder_path+name+"/best_personalized_explanation.txt", "w") as file:
            file.write(json.dumps(best_personalized_explanation))

        best_explanation.update(best_personalized_explanation)
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
    folder_path = "./result/explanation/{}/exp_disco_personalization/{}/iter=".format(group, str(user_id)) + str(iter_count) + "/sample=" + str(sample_num) + "/"
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
        [pool.apply_async(handle, args=(idx, iter_count, folder_path, error_index, error_name, user_id,), callback=update, error_callback=print_error) for idx in target_idx_list]
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
            [pool.apply_async(handle, args=(idx, iter_count, folder_path, error_index, error_name, user_id,), callback=update, error_callback=print_error) for idx in temp_error_index]
            pool.close()
            pool.join()
            print("Error Name: ", error_name)
            print("Error Index: ", error_index)
    
    if mode == "evaluate":
        # Evaluation Global Variables
        evaluation_folder_path = "./result/evaluation/{}/k=".format(group) + str(evaluation_k) + "/exp_disco_personalization/{}/".format(str(user_id)) + "iter=" + str(iter_count) + "/sample=" + str(sample_num) + "/"
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
